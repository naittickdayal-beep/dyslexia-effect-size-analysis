"""Read-only extraction and full available-cohort statistical analysis.

Run with Python + NumPy + SciPy. Source archives are never modified.
Optional --deps supplies a task-local Python dependency directory.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tarfile
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='C:/dyslexia_project')
parser.add_argument('--metadata', default='work')
parser.add_argument('--output', default='outputs')
parser.add_argument('--deps')
args = parser.parse_args()
if args.deps:
    sys.path.insert(0, str(Path(args.deps).resolve()))
import numpy as np
import scipy
from scipy import stats

root, out = Path(args.root), Path(args.output)
out.mkdir(parents=True, exist_ok=True)
datasets = ['ds003126', 'ds005577']
metadata = {d: {r['participant_id']: r for r in csv.DictReader(
    open(Path(args.metadata) / f'{d}-participants.tsv'), delimiter='\t')}
    for d in datasets}
roi_map = {'parstriangularis': 'pars_triangularis_mm',
           'fusiform': 'fusiform_mm', 'insula': 'insula_mm'}
records = collections.defaultdict(list)
audit = []
source_state = {}

def parse_stats(text):
    header = next((l.split()[2:] for l in text.splitlines()
                   if l.startswith('# ColHeaders')), [])
    if 'ThickAvg' not in header:
        return {}
    values = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        row = dict(zip(header, line.split()))
        if row.get('StructName') in roi_map:
            key = roi_map[row['StructName']]
            if key in values:
                raise ValueError('Duplicate region')
            values[key] = float(row['ThickAvg'])
    return values

for path in sorted(root.rglob('*')):
    if not path.is_file() or not path.name.endswith(('.zip', '.tar.gz')):
        continue
    match = re.match(r'(sub-[^_]+)', path.name)
    if not match:
        continue
    sid = match[1]
    matches = [d for d in datasets if sid in metadata[d]]
    if not matches or 'GPU_SEG_ONLY' in path.name:
        continue
    assert len(matches) == 1
    d = matches[0]
    source_state[str(path)] = (path.stat().st_size, path.stat().st_mtime_ns)
    texts = {}
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.endswith('/lh.aparc.DKTatlas.mapped.stats'):
                    texts[n] = z.read(n)
    else:
        with tarfile.open(path, 'r|gz') as t:
            for m in t:
                if m.isfile() and m.name.endswith('/lh.aparc.DKTatlas.mapped.stats'):
                    texts[m.name] = t.extractfile(m).read()
    if not texts:
        audit.append(dict(dataset=d, participant_id=sid, archive=path.name,
                          status='No matching DKT left-hemisphere stats; archive not used'))
    for name, content in texts.items():
        values = parse_stats(content.decode('utf-8', 'replace'))
        if len(values) != 3 or not all(math.isfinite(v) and v > 0 for v in values.values()):
            raise ValueError(f'Invalid primary measurements: {path} {name}')
        group_raw = metadata[d][sid]['group']
        group = {'DL': 'dyslexia', 'TD': 'control'}.get(group_raw, group_raw)
        if group not in ('dyslexia', 'control'):
            continue
        records[(d, sid)].append(dict(dataset=d, participant_id=sid, group=group,
            age=int(metadata[d][sid]['age']), sex=metadata[d][sid]['sex'],
            **values, source_archive=str(path), source_stats=name,
            stats_sha256=hashlib.sha256(content).hexdigest(),
            label_source=f'https://raw.githubusercontent.com/OpenNeuroDatasets/{d}/master/participants.tsv'))

subjects = []
for (d, sid), candidates in sorted(records.items()):
    candidates.sort(key=lambda r: (' (1)' in r['source_archive'], r['source_archive']))
    row = candidates[0].copy()
    for other in candidates[1:]:
        assert all(row[k] == other[k] for k in roi_map.values()), f'Conflicting thickness: {sid}'
        audit.append(dict(dataset=d, participant_id=sid, archive=Path(other['source_archive']).name,
                          status='Duplicate subject: all three ROI values agree; counted once'))
    row['network_ratio'] = row['pars_triangularis_mm'] / math.sqrt(row['fusiform_mm'] * row['insula_mm'])
    row['statistical_exclusion'] = 'none'
    row['qc_status'] = 'Complete positive measurements; visual surface QC not performed'
    subjects.append(row)

def holm(ps):
    order = np.argsort(ps)
    result = np.empty(len(ps))
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, min(1., (len(ps) - rank) * ps[index]))
        result[index] = running
    return result.tolist()

rng = np.random.default_rng(20260919)
results, welch = [], []
features = list(roi_map.values()) + ['network_ratio']
for d in datasets:
    rows = [r for r in subjects if r['dataset'] == d]
    for feature in features:
        x = np.array([r[feature] for r in rows if r['group'] == 'dyslexia'])
        y = np.array([r[feature] for r in rows if r['group'] == 'control'])
        test = stats.mannwhitneyu(x, y, alternative='two-sided', method='asymptotic', use_continuity=True)
        # Independent pair-count check for direction and ties.
        u_pairs = float(np.sum(x[:, None] > y) + .5 * np.sum(x[:, None] == y))
        assert math.isclose(test.statistic, u_pairs, abs_tol=1e-10)
        rb = 2 * test.statistic / (len(x)*len(y)) - 1
        bx = rng.choice(x, (10000, len(x)), replace=True)
        by = rng.choice(y, (10000, len(y)), replace=True)
        bu = stats.mannwhitneyu(bx, by, axis=1, method='asymptotic').statistic
        ci = np.quantile(2 * bu / (len(x)*len(y)) - 1, [.025, .975])
        record = dict(dataset=d, cohort='Pediatric' if d == 'ds003126' else 'Adult',
            measurement=feature, units='dimensionless' if feature=='network_ratio' else 'mm',
            n_dyslexia=len(x), n_control=len(y), U_dyslexia=float(test.statistic),
            p_two_sided=float(test.pvalue), rank_biserial=float(rb),
            rank_biserial_CI_low=float(ci[0]), rank_biserial_CI_high=float(ci[1]))
        for group, a in [('dyslexia',x),('control',y)]:
            q1, med, q3 = np.quantile(a, [.25,.5,.75], method='linear')
            record.update({group+'_median':float(med), group+'_Q1':float(q1),
                group+'_Q3':float(q3), group+'_IQR':float(q3-q1)})
        results.append(record)
        if feature == 'network_ratio':
            t = stats.ttest_ind(x, y, equal_var=False, alternative='two-sided')
            ci_t = t.confidence_interval(.95)
            welch.append(dict(dataset=d, n_dyslexia=len(x), n_control=len(y),
                t=float(t.statistic), df=float(t.df), p_two_sided=float(t.pvalue),
                mean_difference_dyslexia_minus_control=float(x.mean()-y.mean()),
                CI_low=float(ci_t.low), CI_high=float(ci_t.high)))
for r,p in zip(results,holm([r['p_two_sided'] for r in results])):
    r['p_Holm_8_tests'] = p
for r,p in zip(welch,holm([r['p_two_sided'] for r in welch])):
    r['p_Holm_2_tests'] = p

def write_csv(name, rows):
    with open(out/name, 'w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

write_csv('full-cohort-comparisons.csv', results)
write_csv('full-cohort-subject-measurements.csv', subjects)
write_csv('full-cohort-ratio-welch-tests.csv', welch)
write_csv('archive-resolution-audit.csv', audit)
assert all((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==v for p,v in source_state.items())
payload = dict(results=results, subjects=subjects, welch=welch, audit=audit,
    numpy_version=np.__version__, scipy_version=scipy.__version__, seed=20260919,
    bootstrap_resamples=10000, statistical_exclusions=[], source_archives_unchanged=True)
(out/'full-cohort-results.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
print(json.dumps(dict(results=results,welch=welch,audit=audit),indent=2))

