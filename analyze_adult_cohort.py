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
parser.add_argument("--root", default="C:/dyslexia_project")
parser.add_argument("--metadata", default="work")
parser.add_argument("--output", default="outputs")
parser.add_argument("--deps")
args = parser.parse_args()

if args.deps:
    sys.path.insert(0, str(Path(args.deps).resolve()))

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats

root = Path(args.root)
metadata_dir = Path(args.metadata)
output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

dataset = "ds005577"

with open(metadata_dir / f"{dataset}-participants.tsv", encoding="utf-8") as f:
    metadata = {
        row["participant_id"]: row
        for row in csv.DictReader(f, delimiter="\t")
    }

roi_names = {
    "parstriangularis": "pars_triangularis_mm",
    "fusiform": "fusiform_mm",
    "insula": "insula_mm"
}

subject_files = collections.defaultdict(list)
audit = []
source_state = {}


def read_stats(text):
    header = []

    for line in text.splitlines():
        if line.startswith("# ColHeaders"):
            header = line.split()[2:]
            break

    if "ThickAvg" not in header:
        return {}

    values = {}

    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue

        row = dict(zip(header, line.split()))
        region = row.get("StructName")

        if region in roi_names:
            key = roi_names[region]

            if key in values:
                raise ValueError(f"Duplicate region: {region}")

            values[key] = float(row["ThickAvg"])

    return values


for path in sorted(root.rglob("*")):
    if not path.is_file():
        continue

    if not path.name.endswith((".zip", ".tar.gz")):
        continue

    match = re.match(r"(sub-[^_]+)", path.name)

    if not match:
        continue

    subject_id = match.group(1)

    if subject_id not in metadata:
        continue

    if "GPU_SEG_ONLY" in path.name:
        continue

    source_state[str(path)] = (
        path.stat().st_size,
        path.stat().st_mtime_ns
    )

    stats_files = {}

    if path.name.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.endswith("/lh.aparc.DKTatlas.mapped.stats"):
                    stats_files[name] = archive.read(name)

    else:
        with tarfile.open(path, "r|gz") as archive:
            for member in archive:
                if (
                    member.isfile()
                    and member.name.endswith(
                        "/lh.aparc.DKTatlas.mapped.stats"
                    )
                ):
                    file_obj = archive.extractfile(member)

                    if file_obj is not None:
                        stats_files[member.name] = file_obj.read()

    if not stats_files:
        audit.append({
            "participant_id": subject_id,
            "archive": path.name,
            "status": "No usable left-hemisphere DKT statistics found"
        })
        continue

    for stats_name, content in stats_files.items():
        values = read_stats(
            content.decode("utf-8", errors="replace")
        )

        if len(values) != 3:
            raise ValueError(
                f"Missing cortical thickness measurement: "
                f"{path.name}, {stats_name}"
            )

        if not all(
            math.isfinite(value) and value > 0
            for value in values.values()
        ):
            raise ValueError(
                f"Invalid cortical thickness measurement: "
                f"{path.name}, {stats_name}"
            )

        group_raw = metadata[subject_id]["group"]

        group = {
            "DL": "dyslexia",
            "TD": "control"
        }.get(group_raw, group_raw)

        if group not in ("dyslexia", "control"):
            continue

        age_value = metadata[subject_id].get("age", "")

        try:
            age = float(age_value)
        except ValueError:
            age = age_value

        subject_files[subject_id].append({
            "participant_id": subject_id,
            "group": group,
            "age": age,
            "sex": metadata[subject_id].get("sex", ""),
            **values,
            "source_archive": str(path),
            "source_stats": stats_name,
            "stats_sha256": hashlib.sha256(content).hexdigest(),
            "label_source":
                f"https://raw.githubusercontent.com/"
                f"OpenNeuroDatasets/{dataset}/master/participants.tsv"
        })


subjects = []

for subject_id, candidates in sorted(subject_files.items()):
    candidates.sort(
        key=lambda row: (
            " (1)" in row["source_archive"],
            row["source_archive"]
        )
    )

    subject = candidates[0].copy()

    for duplicate in candidates[1:]:
        same_values = all(
            subject[key] == duplicate[key]
            for key in roi_names.values()
        )

        if not same_values:
            raise ValueError(
                f"Conflicting cortical thickness values for {subject_id}"
            )

        audit.append({
            "participant_id": subject_id,
            "archive": Path(
                duplicate["source_archive"]
            ).name,
            "status": "Duplicate subject with matching ROI values"
        })

    subject["network_ratio"] = (
        subject["pars_triangularis_mm"]
        / math.sqrt(
            subject["fusiform_mm"]
            * subject["insula_mm"]
        )
    )

    subjects.append(subject)


n_dyslexia = sum(
    subject["group"] == "dyslexia"
    for subject in subjects
)

n_control = sum(
    subject["group"] == "control"
    for subject in subjects
)

if len(subjects) != 67:
    raise ValueError(
        f"Expected 67 adult subjects, found {len(subjects)}"
    )

if n_dyslexia != 32:
    raise ValueError(
        f"Expected 32 dyslexic adults, found {n_dyslexia}"
    )

if n_control != 35:
    raise ValueError(
        f"Expected 35 control adults, found {n_control}"
    )


def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    previous = 0.0

    for rank, index in enumerate(order):
        value = min(
            1.0,
            (len(p_values) - rank) * p_values[index]
        )

        previous = max(previous, value)
        adjusted[index] = previous

    return adjusted.tolist()


rng = np.random.default_rng(20260919)

features = [
    "pars_triangularis_mm",
    "fusiform_mm",
    "insula_mm",
    "network_ratio"
]

results = []

for feature in features:
    dyslexia = np.array([
        subject[feature]
        for subject in subjects
        if subject["group"] == "dyslexia"
    ])

    control = np.array([
        subject[feature]
        for subject in subjects
        if subject["group"] == "control"
    ])

    test = stats.mannwhitneyu(
        dyslexia,
        control,
        alternative="two-sided",
        method="asymptotic",
        use_continuity=True
    )

    u_check = float(
        np.sum(dyslexia[:, None] > control)
        + 0.5 * np.sum(dyslexia[:, None] == control)
    )

    if not math.isclose(
        test.statistic,
        u_check,
        abs_tol=1e-10
    ):
        raise ValueError(
            f"Mann-Whitney U check failed for {feature}"
        )

    rank_biserial = (
        2 * test.statistic
        / (len(dyslexia) * len(control))
        - 1
    )

    bootstrap_dyslexia = rng.choice(
        dyslexia,
        size=(10000, len(dyslexia)),
        replace=True
    )

    bootstrap_control = rng.choice(
        control,
        size=(10000, len(control)),
        replace=True
    )

    bootstrap_u = stats.mannwhitneyu(
        bootstrap_dyslexia,
        bootstrap_control,
        axis=1,
        method="asymptotic"
    ).statistic

    bootstrap_effects = (
        2 * bootstrap_u
        / (len(dyslexia) * len(control))
        - 1
    )

    ci_low, ci_high = np.quantile(
        bootstrap_effects,
        [0.025, 0.975]
    )

    dyslexia_q1, dyslexia_median, dyslexia_q3 = np.quantile(
        dyslexia,
        [0.25, 0.5, 0.75]
    )

    control_q1, control_median, control_q3 = np.quantile(
        control,
        [0.25, 0.5, 0.75]
    )

    results.append({
        "measurement": feature,
        "units":
            "dimensionless"
            if feature == "network_ratio"
            else "mm",
        "n_dyslexia": len(dyslexia),
        "n_control": len(control),
        "U_dyslexia": float(test.statistic),
        "p_two_sided": float(test.pvalue),
        "rank_biserial": float(rank_biserial),
        "rank_biserial_CI_low": float(ci_low),
        "rank_biserial_CI_high": float(ci_high),
        "dyslexia_median": float(dyslexia_median),
        "dyslexia_Q1": float(dyslexia_q1),
        "dyslexia_Q3": float(dyslexia_q3),
        "dyslexia_IQR": float(
            dyslexia_q3 - dyslexia_q1
        ),
        "control_median": float(control_median),
        "control_Q1": float(control_q1),
        "control_Q3": float(control_q3),
        "control_IQR": float(
            control_q3 - control_q1
        )
    })


adjusted = holm_adjust([
    result["p_two_sided"]
    for result in results
])

for result, p_value in zip(results, adjusted):
    result["p_Holm_4_tests"] = p_value


ratio_dyslexia = np.array([
    subject["network_ratio"]
    for subject in subjects
    if subject["group"] == "dyslexia"
])

ratio_control = np.array([
    subject["network_ratio"]
    for subject in subjects
    if subject["group"] == "control"
])

welch_test = stats.ttest_ind(
    ratio_dyslexia,
    ratio_control,
    equal_var=False,
    alternative="two-sided"
)

welch_ci = welch_test.confidence_interval(0.95)

welch = [{
    "n_dyslexia": len(ratio_dyslexia),
    "n_control": len(ratio_control),
    "t": float(welch_test.statistic),
    "df": float(welch_test.df),
    "p_two_sided": float(welch_test.pvalue),
    "mean_difference_dyslexia_minus_control":
        float(
            ratio_dyslexia.mean()
            - ratio_control.mean()
        ),
    "CI_low": float(welch_ci.low),
    "CI_high": float(welch_ci.high)
}]


def write_csv(filename, rows):
    if not rows:
        return

    with open(
        output_dir / filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


write_csv(
    "adult-statistical-results.csv",
    results
)

write_csv(
    "adult-subject-measurements.csv",
    subjects
)

write_csv(
    "adult-ratio-welch-test.csv",
    welch
)

write_csv(
    "adult-archive-audit.csv",
    audit
)


result_lookup = {
    result["measurement"]: result
    for result in results
}

figure_features = [
    (
        "pars_triangularis_mm",
        "A. Pars triangularis",
        "Cortical thickness (mm)"
    ),
    (
        "fusiform_mm",
        "B. Fusiform",
        "Cortical thickness (mm)"
    ),
    (
        "insula_mm",
        "C. Insula",
        "Cortical thickness (mm)"
    ),
    (
        "network_ratio",
        "D. Network ratio",
        "Network ratio"
    )
]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 9)
)

axes = axes.flatten()

for ax, (feature, title, ylabel) in zip(
    axes,
    figure_features
):
    control = np.array([
        subject[feature]
        for subject in subjects
        if subject["group"] == "control"
    ])

    dyslexia = np.array([
        subject[feature]
        for subject in subjects
        if subject["group"] == "dyslexia"
    ])

    ax.boxplot(
        [control, dyslexia],
        positions=[1, 2],
        widths=0.46,
        showfliers=False,
        medianprops={"linewidth": 1.7},
        boxprops={"linewidth": 1.2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2}
    )

    control_jitter = np.linspace(
        -0.08,
        0.08,
        len(control)
    )

    dyslexia_jitter = np.linspace(
        -0.08,
        0.08,
        len(dyslexia)
    )

    ax.scatter(
        np.ones(len(control)) + control_jitter,
        control,
        s=26,
        alpha=0.7,
        marker="o",
        zorder=3
    )

    ax.scatter(
        np.ones(len(dyslexia)) * 2 + dyslexia_jitter,
        dyslexia,
        s=26,
        alpha=0.7,
        marker="s",
        zorder=3
    )

    ax.set_xticks([1, 2])
    ax.set_xticklabels([
        "Control",
        "Dyslexia"
    ])

    ax.set_ylabel(ylabel)

    ax.set_title(
        title,
        loc="left",
        fontweight="bold"
    )

    result = result_lookup[feature]

    p_value = result["p_two_sided"]
    effect = result["rank_biserial"]

    if p_value < 0.0001:
        p_text = "p < 0.0001"
    else:
        p_text = f"p = {p_value:.4f}"

    ax.text(
        0.97,
        0.97,
        f"{p_text}\n"
        + r"$r_{rb}$"
        + f" = {effect:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(
        axis="y",
        alpha=0.2
    )


fig.tight_layout(
    h_pad=2.5,
    w_pad=2.5
)

figure_path = (
    output_dir
    / "Figure2_Adult_Comparisons.png"
)

fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


for path, original_state in source_state.items():
    current_state = (
        Path(path).stat().st_size,
        Path(path).stat().st_mtime_ns
    )

    if current_state != original_state:
        raise RuntimeError(
            f"Source archive changed: {path}"
        )


payload = {
    "dataset": dataset,
    "cohort": {
        "total": len(subjects),
        "dyslexia": n_dyslexia,
        "control": n_control
    },
    "results": results,
    "subjects": subjects,
    "welch": welch,
    "audit": audit,
    "numpy_version": np.__version__,
    "scipy_version": scipy.__version__,
    "seed": 20260919,
    "bootstrap_resamples": 10000,
    "source_archives_unchanged": True
}

(
    output_dir
    / "adult-analysis-results.json"
).write_text(
    json.dumps(payload, indent=2),
    encoding="utf-8"
)


print(
    f"Adult cohort: "
    f"{n_dyslexia} dyslexic, "
    f"{n_control} control, "
    f"{len(subjects)} total"
)

print(
    f"Figure saved to "
    f"{figure_path}"
)

print(
    json.dumps(
        results,
        indent=2
    )
)
