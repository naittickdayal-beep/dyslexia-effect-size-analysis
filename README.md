# Dyslexia Cortical-Thickness Analysis

This repository contains the analysis code and results used to compare cortical thickness measurements between dyslexic and typically developing participants in pediatric and adult datasets.

The analysis includes:

- a 35-subject pediatric cohort from ds003126 (17 dyslexic, 18 typically developing controls)
- a 67-subject adult cohort from ds005577 (32 dyslexic, 35 typically developing controls)

The network-based cortical thickness ratio was calculated as:

pars_triangularis / sqrt(fusiform * insula)

Group differences were tested using two-sided Mann–Whitney U tests. Rank-biserial correlations were used to measure effect size, and bootstrap confidence intervals were calculated to estimate uncertainty.

The scripts use previously extracted cortical thickness measurements and do not modify the original MRI data or processed subject files.

## Files

- analyze_pediatric_35.py: analyzes the 35-subject pediatric cohort
- analyze_adult_67.py: analyzes the 67-subject adult cohort
- pediatric_subject_measurements.csv: contains pediatric subject-level cortical thickness measurements
- pediatric_statistical_results.csv: contains pediatric statistical comparison results
- adult_subject_measurements.csv: contains adult subject-level cortical thickness measurements
- adult_statistical_results.csv: contains adult statistical comparison results
