# PAM50 Breast Cancer Subtype Classification

Semi-supervised representation learning for PAM50 subtype classification from TCGA-BRCA RNA-seq data.

## Research Question

**Can semi-supervised representation learning from bulk RNA-seq data improve the reliability of PAM50 subtype classification when labeled samples are extremely limited, compared with a supervised-only approach?**

---

## Dataset

| Item | Value |
| --- | --- |
| Expression matrix | HiSeqV2.csv — 1,218 samples x 20,530 genes (log2-scale) |
| Labeled samples | 69 (PAM50 labels, Unknown/Normal excluded) |
| Unlabeled samples | 1,149 (expression only) |
| p/n ratio | ~298x (extreme high-dimensionality) |
| Classes | Basal (15), Her2 (11), LumA (18), LumB (25) |

---

## Method

**Controlled comparison:** NB03 and NB04 are identical except for what data PCA is fitted on.

| | NB03 — Supervised Baseline | NB04 — Semi-supervised |
| --- | --- | --- |
| PCA fitted on | Labeled training fold only (~55 samples) | Labeled training fold + all unlabeled (~1,204 samples) |
| Preprocessing | VarianceThreshold + StandardScaler | identical |
| Classifier | ElasticNet LogisticRegression (l1_ratio tunable) | identical |
| Evaluation | Nested 5x3-fold CV | identical |

PCA is unsupervised — it never uses labels — so including unlabeled data for PCA fitting is valid.

**N_COMPONENTS selection:** `min(n_80_variance_full, n_labeled // 3) = min(69, 23) = 23`

---

## Results

| Pipeline | Macro F1 (mean +/- SD) | Accuracy |
| --- | --- | --- |
| NB03 — Supervised baseline | 0.699 +/- 0.117 | 0.711 |
| NB04 — Semi-supervised | **0.804 +/- 0.079** | 0.799 |
| Delta | **+0.105** | +0.088 |

**Per-class F1:**

| Subtype | Baseline | Semi-supervised |
| --- | --- | --- |
| Basal | 0.903 | 0.968 |
| Her2 | 0.609 | 0.720 |
| LumA | 0.773 | 0.842 |
| LumB | 0.550 | 0.682 |

Semi-supervised PCA improves all 4 subtypes. Variance also drops (0.117 -> 0.079), meaning more stable fold-to-fold performance.

---

## Run Order

```text
NB01 -> NB02 -> NB03 -> NB04 -> NB05
```

| Notebook | Purpose | Outputs |
| --- | --- | --- |
| 01_data_understanding | EDA, labeled/unlabeled split | 01_dataset_summary.json |
| 02_gene_structure | Variance, correlation, PCA stability, N_COMPONENTS selection | 02_n_components.json |
| 03_supervised_baseline | Nested CV, PCA on labeled only | 03_baseline_summary.json, 03_baseline_cv_results.csv |
| 04_semisupervised_pipeline | Nested CV, PCA on labeled+unlabeled | 04_ssl_summary.json, 04_ssl_cv_results.csv |
| 05_results_discussion | Comparison, per-class F1, PCA loadings, conclusions | figures |

NB03 and NB04 both require `02_n_components.json` to exist (produced by NB02).

---

## Project Structure

```text
.
├── data/
│   ├── HiSeqV2.csv              # expression matrix
│   └── brca_pam50.csv           # PAM50 labels
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_gene_structure.ipynb
│   ├── 03_supervised_baseline.ipynb
│   ├── 04_semisupervised_pipeline.ipynb
│   └── 05_results_discussion.ipynb
├── reports/
│   ├── figures/                 # generated plots
│   └── tables/                  # generated JSON/CSV
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

Run notebooks 01 through 05 in order. Each notebook is self-contained and re-loads data from scratch.
