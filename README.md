# PAM50 Breast Cancer Subtype Classification

A reproducible machine learning pipeline for classifying breast cancer molecular subtypes (PAM50) using TCGA-BRCA RNA-Seq gene expression data. This project demonstrates proper handling of high-dimensional genomic data with class imbalance, featuring interpretable models and robust cross-validation.

## Project Overview

Breast cancer is a heterogeneous disease with distinct molecular subtypes that have different prognoses and treatment responses. The PAM50 classifier identifies five intrinsic subtypes:

- **Luminal A (LumA)**: Best prognosis, hormone receptor positive
- **Luminal B (LumB)**: Moderate prognosis, more aggressive than LumA
- **HER2-enriched (Her2)**: Targetable with HER2 therapies
- **Basal-like (Basal)**: Most aggressive, often triple-negative
- **Normal-like (Normal)**: Similar to normal breast tissue

This project builds an interpretable machine learning model to classify samples into these subtypes based on gene expression profiles.

---

## Data Summary

| Metric | Count |
|--------|-------|
| Expression samples (HiSeqV2.csv) | 1,218 |
| PAM50 raw label rows | 141 |
| Unique samples after deduplication | 74 |
| Unknown removed | 4 |
| **Labeled samples after dedup + exclusion** | **70** |
| Normal excluded (< 2 samples for CV) | 1 |
| **Supervised training set** | **69** |
| Unlabeled samples (expression only) | 1,148 |

### Data Processing

The PAM50 labels file undergoes:
1. **TCGA barcode normalization**: All IDs truncated to 15 chars for matching
2. **Deduplication**: Duplicate sample IDs removed (keep first), 141 rows → 74 unique
3. **Exclusion**: Samples with "Unknown" subtype are removed (4 samples)
4. **Rare-class exclusion**: Normal-like (1 sample) excluded from supervised training

After processing, **70 labeled samples** remain (69 used in supervised training after Normal exclusion).

### Label Distribution (Training Set, n=69)

| Subtype | Count | % |
|---------|-------|---|
| LumB | 25 | 36.2% |
| LumA | 18 | 26.1% |
| Basal | 15 | 21.7% |
| Her2 | 11 | 15.9% |

Class imbalance ratio (LumB/Her2): 2.3× — handled via Stratified CV and `class_weight='balanced'`.

---

## Task Definition

### Input
- **X**: Gene expression vector per sample (RNA-Seq, 20,530 genes)
- **y**: PAM50 subtype label (one of: LumA, LumB, Her2, Basal)

### Pipeline (inside cross-validation)
1. `Log2Transformer` — log₂(x+1) if data is raw counts
2. `VarianceThreshold(0.01)` — remove near-constant genes
3. `StandardScaler` — zero-mean, unit-variance
4. `SelectKBest(f_classif, k)` — ANOVA F-test feature selection
5. `LogisticRegression(penalty='elasticnet', solver='saga', class_weight='balanced')`

### Output
- **Predicted Class**: One of {LumA, LumB, Her2, Basal}
- **Evaluation Metrics**: Macro-F1 (primary), accuracy, per-class precision/recall/F1
- **Top Genes**: Coefficients identifying discriminative genes per subtype

---

## Processing Rules (Explicit, No Auto-Magic)

### 1. Orientation Detection
```
IF first 10 column names start with "TCGA-" AND first 10 row names do NOT:
    -> Genes are rows, samples are columns -> TRANSPOSE
ELSE IF first 10 row names start with "TCGA-":
    -> Samples are rows, genes are columns -> NO TRANSPOSE
ELSE:
    -> Fallback: if rows > columns, assume genes as rows -> TRANSPOSE
```

### 2. Sample ID Normalization
```
TCGA Barcode Levels:
  - Patient:  TCGA-XX-XXXX      (12 chars)
  - Sample:   TCGA-XX-XXXX-01   (15 chars) <- WE USE THIS
  - Vial:     TCGA-XX-XXXX-01A  (16 chars)
  - Full:     TCGA-XX-XXXX-01A-11R-A13K-07

RULE: Truncate all IDs to 15 characters (sample level)
  - Expression: TCGA-AR-A5QQ-01 (already 15) -> TCGA-AR-A5QQ-01
  - PAM50:      TCGA-A7-A13F-01A (16 chars)  -> TCGA-A7-A13F-01
```

### 3. Log Transform Decision
```
IF max(expression_values) > 100:
    -> Likely raw counts -> Apply log2(x+1)
ELSE:
    -> Already normalized -> No transform

Current data: max ~20 -> No additional log transform applied
```

### 4. Excluded Labels
```
Exclude samples where PAM50 label is in:
    ["Unknown", "unknown", "NA", "N/A", "", None]
```

---

## Unlabeled Samples Handling

**Critical**: Unlabeled samples (1,148 total) are **NOT** used in:
- Training
- Cross-validation
- Model selection
- Evaluation metrics

**Inference Demo** (NB05):
- The saved model is applied to unlabeled samples (inference only)
- Predictions saved to `reports/tables/unlabeled_predictions.csv`
- High-confidence pseudo-labels selected by proba ≥ 0.95, margin ≥ 0.20
- These predictions are **NOT VALIDATED** (no ground truth exists)

---

## Repository Structure

```
.
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment
├── app.py                       # Streamlit dashboard application
│
├── data/                        # Data files (place here)
│   ├── HiSeqV2.csv             # Expression matrix (20,530 genes × 1,218 samples)
│   ├── brca_pam50.csv.csv      # PAM50 labels (141 rows -> 74 unique -> 70 after Unknown)
│   └── TCGA-BRCA.clinical.tsv  # Clinical metadata (optional, for enrichment)
│
├── notebooks/                   # Self-contained analysis notebooks (primary workflow)
│   ├── 01_data_audit.ipynb     # Data quality checks, label distribution, distributions
│   ├── 02_baseline_model.ipynb # ElasticNet training, CV evaluation, permutation test
│   ├── 03_model_comparison.ipynb  # ElasticNet vs SVM vs RandomForest comparison
│   ├── 04_feature_insights.ipynb  # Coefficients, stability, Kruskal-Wallis tests
│   └── 05_unlabeled_inference.ipynb  # Unlabeled predictions + clinical enrichment
│
├── src/                         # Dashboard package only
│   └── dashboard/
│       ├── data_loader.py       # Data loading for dashboard
│       └── plots.py             # Plotly visualization helpers
│
└── reports/                     # Generated outputs (produced by notebooks)
    ├── elasticnet_pipeline.pkl  # Saved model (produced by NB02)
    ├── figures/                 # Plots (PNG)
    └── tables/                  # CSV/JSON summaries
```

---

## Quick Start

### 1. Setup Environment

```bash
# Using pip
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# OR using conda
conda env create -f environment.yml
conda activate pam50
```

### 2. Place Data Files

Put data files in `data/` directory:
- `HiSeqV2.csv` (expression matrix)
- `brca_pam50.csv.csv` (PAM50 labels)
- `TCGA-BRCA.clinical.tsv` (optional, for NB05 clinical enrichment)

### 3. Run Notebooks in Order

```bash
jupyter notebook notebooks/
```

Run notebooks **01 through 05 in sequence**. Each notebook is fully self-contained — no imports from `src/`. NB05 depends on the model pickle saved by NB02.

| Notebook | Output | Time |
|----------|--------|------|
| 01_data_audit | data_truth_table.json, figures | ~30s |
| 02_baseline_model | model_metrics.csv, confusion_matrix.png, elasticnet_pipeline.pkl | ~2min |
| 03_model_comparison | 03_model_comparison.csv | ~5min |
| 04_feature_insights | top_genes_all_classes.csv, coefficient heatmap | ~2min |
| 05_unlabeled_inference | unlabeled_predictions.csv, pseudolabels_*.csv | ~1min |

### 4. Launch Dashboard

```bash
streamlit run app.py
```

The dashboard reads all outputs from `reports/`. Run the notebooks first to generate them.

---

## Dashboard

The Streamlit dashboard provides interactive exploration of PAM50 predictions.

**Features:**
- **Sample Explorer**: Browse individual predictions with confidence scores
- **Probability Visualization**: Interactive bar charts of class probabilities
- **Gene Explanations**: Top contributing genes (coefficients) per subtype
- **Clinical Context**: Patient clinical data when available
- **Model Performance**: CV metrics and confusion matrix
- **Data Notes**: Sample provenance and data quality statistics

**Key files read by dashboard:**

| File | Produced by |
|------|------------|
| `reports/tables/data_truth_table.json` | NB01 |
| `reports/tables/model_metrics.csv` | NB02 |
| `reports/tables/classification_report.csv` | NB02 |
| `reports/figures/confusion_matrix.png` | NB02 |
| `reports/elasticnet_pipeline.pkl` | NB02 |
| `reports/tables/top_genes_all_classes.csv` | NB04 |
| `reports/tables/unlabeled_predictions.csv` | NB05 |
| `reports/tables/pseudolabels_*.csv/json` | NB05 |

**Important**: This is a demonstration tool — NOT for clinical use.

---

## Output Artifacts

### Figures (`reports/figures/`)
| File | Description |
|------|-------------|
| `01_label_distribution.png` | Class distribution (training set) |
| `01_expression_distributions.png` | Expression value and variance distributions |
| `confusion_matrix.png` | Normalized confusion matrix (CV hold-out) |
| `04_coefficient_heatmap.png` | Gene coefficients per class (ElasticNet) |
| `04_stability_analysis.png` | Feature selection stability across CV folds |
| `03_model_comparison_f1.png` | Macro-F1 comparison across models |
| `05_prediction_distribution.png` | Unlabeled sample prediction distribution |

### Tables (`reports/tables/`)
| File | Description |
|------|-------------|
| `data_truth_table.json` | Sample provenance counts (dashboard reads this) |
| `model_metrics.csv` | Macro-F1, accuracy, train/test/shuffled |
| `classification_report.csv` | Per-class precision, recall, F1 |
| `top_genes_all_classes.csv` | Top discriminative genes per subtype |
| `04_kruskal_wallis_results.csv` | Statistical tests on top genes |
| `unlabeled_predictions.csv` | All predictions on unlabeled samples |
| `pseudolabels_high_confidence.csv` | High-confidence pseudo-labels |
| `pseudolabels_with_clinical.csv` | Pseudo-labels enriched with clinical data |
| `pseudolabels_summary.json` | Selection statistics |

---

## Methodology

### Why ElasticNet?

| Challenge | Solution |
|-----------|----------|
| p >> n (20,530 genes vs 69 training samples) | L1 regularization for sparse feature selection |
| Multicollinearity (co-expressed genes) | L2 regularization for coefficient stability |
| Need interpretability | Direct coefficients per gene per class |
| Class imbalance (2.3× ratio) | `class_weight='balanced'`, Macro-F1 |

### Cross-Validation Strategy

- **Method**: Stratified 5-Fold CV (all preprocessing inside the fold)
- **No held-out test set**: 69 samples is too small; all 69 used for CV
- **Leakage prevention**: VarianceThreshold, StandardScaler, SelectKBest fitted on training fold only
- **Permutation check**: Shuffled-label CV verifies no data leakage

---

## Results Summary

### Model Performance (5-Fold Stratified CV, 69 samples, 4 classes)

| Metric | Score |
|--------|-------|
| **Macro-F1 (CV mean)** | **0.856 ± 0.059** |
| Accuracy | 0.855 |
| Train F1 (overfit check) | 1.000 |
| Shuffled-label F1 (anti-leakage) | 0.322 |

### Per-Class Performance

| Subtype | Precision | Recall | F1 | n |
|---------|-----------|--------|----|---|
| Basal | 1.000 | 0.933 | 0.966 | 15 |
| Her2 | 0.800 | 0.727 | 0.762 | 11 |
| LumA | 0.889 | 0.889 | 0.889 | 18 |
| LumB | 0.778 | 0.840 | 0.808 | 25 |

### Top Discriminative Genes by Subtype

| Subtype | Top Positive Genes | Top Negative Genes |
|---------|-------------------|-------------------|
| Basal | POU5F1, ROPN1B, ROPN1 | SLC4A8, GPR160 |
| Her2 | C9orf152, SIDT1, TSPAN15 | CHODL, ESR1, GABRP |
| LumA | XBP1, C5orf36, DSG1 | CEP55, CENPA, EN1 |
| LumB | RNF103, CENPA, GPR77 | PPP1R14C, DSG1, LOC100127888 |

---

## Reproducibility

- **Random Seed**: 42 (fixed for all stochastic operations)
- **Environment**: Exact versions in `requirements.txt`
- **Notebooks**: Run 01 → 05 in order to regenerate all outputs from scratch

---

## Limitations

1. **High-dimensional data**: 69 training samples vs 20,530 genes (p/n ≈ 298×)
2. **Class imbalance**: Her2 subtype has only 11 samples
3. **No external validation**: Only internal CV on the same TCGA cohort
4. **Unlabeled predictions**: Not validated, demonstration only
5. **Normal subtype excluded**: Only 1 sample — insufficient for stratified CV

---

## Troubleshooting

### "No overlapping samples found"
Sample IDs must be TCGA barcodes. The notebooks normalize to 15 chars. Check that your `HiSeqV2.csv` and `brca_pam50.csv.csv` use TCGA barcode format.

### NB05 pickle load error
NB05 patches `sys.modules` before loading the pickle. If you see an ImportError for `src.preprocessing`, ensure you run the pickle-patch cell before the load cell.

### Dashboard shows missing data
Run all 5 notebooks first (01 → 05) to generate the required files in `reports/`.

---

## Appendix: High-Confidence Pseudo-Labeling (Experimental)

### ⚠ NOT GROUND TRUTH

Pseudo-labels are model predictions, not validated PAM50 subtypes. They should **NEVER** be used for:
- Model evaluation
- Clinical decision-making
- Reporting as "true" labels

### What NB05 Does

1. Load the saved ElasticNet model (`reports/elasticnet_pipeline.pkl`)
2. Predict class probabilities for 1,148 unlabeled samples
3. Select high-confidence predictions: top proba ≥ 0.95 AND margin ≥ 0.20
4. Enrich selected samples with clinical metadata (age, stage, grade)
5. Save outputs to `reports/tables/`

### Why Pseudo-Labels?

Exploratory uses (with caveats):
- Hypothesis generation for future lab studies
- Understanding model behavior on diverse samples
- Descriptive clinical-subtype associations

### Why NOT for Evaluation?

- Selection bias: only confident predictions are retained
- Circular reasoning: cannot validate a model with its own predictions

---

## Citation

```
TCGA-BRCA RNA-Seq data from:
The Cancer Genome Atlas (TCGA) — https://www.cancer.gov/tcga
```

---

## License

Educational purposes. See LICENSE file.
