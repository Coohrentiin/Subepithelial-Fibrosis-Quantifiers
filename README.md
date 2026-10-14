# Quantification of corneal surgery antecedent — reproducibility package

This repository is companion to our paper https://doi.org/10.1371/journal.pone.0328320

Minimal data and code to reproduce every quantitative result and figure of the
manuscript. The analysis works entirely from **anonymised OCT quantifier
tables**: no patient image and no patient identifier are included or required.

## Reproduce the results

```bash
pip install -r requirements.txt
python main.py
```

This regenerates:

- `outputs/figures/` — every figure of the paper (PDF)
- `outputs/csv/` — the minimal data set behind every mean, standard deviation,
  graph and model (flat CSV files)

## Contents

| Path | Description |
|------|-------------|
| `data/*.pkl` | Anonymised OCT quantifier tables (the minimal data set, pickled). |
| `data/train_dataset.csv`, `data/test_dataset.csv` | Anonymised classifier train/test split. |
| `main.py` | Reproduces all statistics, models and figures. |
| `src/` | OCT preprocessing / quantification code used to derive the quantifiers from raw images. |
| `build_data.py` | Provenance only: how the anonymised tables were produced from the raw acquisitions. Reviewers do **not** need to run it. |

## Anonymisation

Each acquisition is identified by an anonymous code in the `Origin` column
(e.g. `P001_OD_Line`). The code encodes only non-identifying structure: an
opaque patient number, the eye side (OD/OS), the scan protocol, and, for the
longitudinal analysis, the post-operative month. Patient names and record
numbers have been removed and cannot be recovered from these files.

## Quantifiers

For each cornea the following OCT-derived quantifiers are provided:

| Column | Meaning |
|--------|---------|
| `Peak Width` | Width of the posterior stromal intensity peak (px). |
| `Sigma` | Standard deviation of the fitted Gaussian (bell curve). |
| `Mean` | Relative depth (position) of the fitted bell curve. |
| `COV` | Covariance between the fitted and measured intensity profile. |
| `MSE` | Mean squared error of the Gaussian fit. |
| `Area Ratio` | Ratio of the bell-curve area to the total profile area. |
| `Alpha`, `Beta` | Parameters of the exponential attenuation fit. |
| `Peak Intensity` | Normalised intensity at the posterior peak. |
| `SNR` | Image signal-to-noise ratio (marker sizing only). |
| `Healthy` | Label: 1 = healthy, 0 = Fuchs dystrophy, 2 = visual haze. |
| `Origin` | Anonymous acquisition code. |

## Requirements

Python 3.10+ and the packages listed in `requirements.txt`.

## Citation 

Soubeiran C, Vilbert M, Memmi B, Georgeon C, Borderie V, Chessel A, et al. (2026) Machine learning quantitative OCT derived fibrosis indicators for corneal haze evaluation. PLoS One 21(10): e0328320. https://doi.org/10.1371/journal.pone.0328320