# Dyslexia cortical-thickness analysis export

This export contains the reproducible analysis code and result files used for:

- the exact 35-subject pediatric cohort (`ds003126`: 17 dyslexia, 18 control),
- the `ds005577` section of the full-cohort analysis.

The code reads subject-level cortical-thickness values from the project’s extracted results. It computes the network ratio as:

```text
pars_triangularis / sqrt(fusiform * insula)
```

The statistical comparisons are two-sided Mann–Whitney U tests with rank-biserial effect sizes and bootstrap confidence intervals. The scripts do not delete or modify source archives.

## Files

- `analyze_pediatric_35.py`: reproduces the specified pediatric cohort analysis.
- `analyze_full_cohort.py`: extracts measurements and computes the full-cohort results; filter `full_cohort_*` to inspect the ds005577 section.
- `pediatric_35_*`: pediatric subjects and results.
- `full_cohort_*`: full extracted subject measurements and comparisons, including ds005577.

The adult full-cohort table contains the available `ds005577` subjects and should be interpreted separately from any 67-subject manuscript-target subset.

