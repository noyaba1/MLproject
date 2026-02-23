import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"

SUBTYPES = ["Basal", "Her2", "LumA", "LumB"]


def normalize_sample_id(sample_id: str, length: int = 15) -> str:
    if pd.isna(sample_id):
        return ""
    return str(sample_id)[:length]


def get_patient_id(sample_id: str) -> str:
    return normalize_sample_id(sample_id, 12)


def detect_probability_columns(df: pd.DataFrame) -> Dict[str, str]:
    prob_cols = {}

    for subtype in SUBTYPES:
        candidates = [
            f"prob_{subtype}",
            f"proba_{subtype}",
            f"Prob_{subtype}",
            f"Proba_{subtype}",
            subtype,
        ]

        for col in candidates:
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    prob_cols[subtype] = col
                    break

    return prob_cols


def load_unlabeled_predictions() -> Tuple[Optional[pd.DataFrame], str]:
    filepath = TABLES_DIR / "unlabeled_predictions.csv"

    if not filepath.exists():
        return None, f"File not found: {filepath.name}"

    try:
        df = pd.read_csv(filepath)

        if "sample_id" not in df.columns:
            id_cols = [c for c in df.columns if "sample" in c.lower() or "id" in c.lower()]
            if id_cols:
                df = df.rename(columns={id_cols[0]: "sample_id"})
            else:
                return None, "No sample_id column found"

        return df, f"Loaded {len(df)} predictions"

    except Exception as e:
        return None, f"Error loading: {str(e)}"


def load_pseudolabels_high_confidence() -> Tuple[Optional[pd.DataFrame], str]:
    filepath = TABLES_DIR / "pseudolabels_high_confidence.csv"

    if not filepath.exists():
        return None, f"File not found: {filepath.name}"

    try:
        df = pd.read_csv(filepath)
        return df, f"Loaded {len(df)} high-confidence pseudo-labels"

    except Exception as e:
        return None, f"Error loading: {str(e)}"


def load_pseudolabels_with_clinical() -> Tuple[Optional[pd.DataFrame], str]:
    filepath = TABLES_DIR / "pseudolabels_with_clinical.csv"

    if not filepath.exists():
        return None, f"File not found: {filepath.name}"

    try:
        df = pd.read_csv(filepath)
        return df, f"Loaded {len(df)} samples with clinical data"

    except Exception as e:
        return None, f"Error loading: {str(e)}"


def load_clinical_data() -> Tuple[Optional[pd.DataFrame], str]:
    filepath = DATA_DIR / "TCGA-BRCA.clinical.tsv"

    if not filepath.exists():
        return None, f"File not found: {filepath.name}"

    try:
        df = pd.read_csv(filepath, sep="\t")

        if "submitter_id" in df.columns:
            df["patient_id"] = df["submitter_id"].apply(lambda x: str(x)[:12] if pd.notna(x) else "")

        return df, f"Loaded {len(df)} clinical records"

    except Exception as e:
        return None, f"Error loading: {str(e)}"


def load_top_genes() -> Tuple[Optional[pd.DataFrame], str]:
    filepath = TABLES_DIR / "top_genes_all_classes.csv"

    if not filepath.exists():
        return None, f"File not found: {filepath.name}"

    try:
        df = pd.read_csv(filepath)
        return df, f"Loaded coefficients for {df['class'].nunique()} classes"

    except Exception as e:
        return None, f"Error loading: {str(e)}"


def load_model_metrics() -> Tuple[Optional[pd.DataFrame], str]:
    filepath = TABLES_DIR / "model_metrics.csv"

    if not filepath.exists():
        return None, f"File not found: {filepath.name}"

    try:
        df = pd.read_csv(filepath)
        return df, "Loaded model metrics"

    except Exception as e:
        return None, f"Error loading: {str(e)}"


def load_classification_report() -> Tuple[Optional[pd.DataFrame], str]:
    filepath = TABLES_DIR / "classification_report.csv"

    if not filepath.exists():
        return None, f"File not found: {filepath.name}"

    try:
        df = pd.read_csv(filepath, index_col=0)
        return df, "Loaded classification report"

    except Exception as e:
        return None, f"Error loading: {str(e)}"


def load_pseudolabels_summary() -> Tuple[Optional[Dict], str]:
    filepath = TABLES_DIR / "pseudolabels_summary.json"

    if not filepath.exists():
        return None, f"File not found: {filepath.name}"

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data, "Loaded pseudo-labeling summary"

    except Exception as e:
        return None, f"Error loading: {str(e)}"


def get_figure_path(name: str) -> Optional[Path]:
    filepath = FIGURES_DIR / name
    if filepath.exists():
        return filepath
    return None


def get_top_genes_for_class(
    genes_df: pd.DataFrame,
    subtype: str,
    n: int = 10,
    direction: str = "positive"
) -> pd.DataFrame:
    if genes_df is None:
        return pd.DataFrame()

    mask = (genes_df["class"] == subtype) & (genes_df["direction"] == direction)
    subset = genes_df[mask].copy()

    subset["abs_coef"] = subset["coefficient"].abs()
    subset = subset.nlargest(n, "abs_coef")

    return subset[["gene", "coefficient"]].reset_index(drop=True)


def get_clinical_for_sample(
    clinical_df: pd.DataFrame,
    sample_id: str
) -> Optional[Dict]:
    if clinical_df is None:
        return None

    patient_id = get_patient_id(sample_id)

    if "patient_id" in clinical_df.columns:
        match = clinical_df[clinical_df["patient_id"] == patient_id]
    elif "submitter_id" in clinical_df.columns:
        match = clinical_df[clinical_df["submitter_id"].str[:12] == patient_id]
    else:
        return None

    if match.empty:
        return None

    row = match.iloc[0]

    clinical_info = {}

    age_cols = [c for c in row.index if "age" in c.lower()]
    for col in age_cols:
        if pd.notna(row[col]):
            clinical_info["Age"] = row[col]
            break

    gender_cols = [c for c in row.index if "gender" in c.lower() or "sex" in c.lower()]
    for col in gender_cols:
        if pd.notna(row[col]):
            clinical_info["Gender"] = row[col]
            break

    stage_cols = [c for c in row.index if "stage" in c.lower() and "pathologic" in c.lower()]
    for col in stage_cols:
        if pd.notna(row[col]):
            clinical_info["Stage"] = row[col]
            break

    grade_cols = [c for c in row.index if "grade" in c.lower()]
    for col in grade_cols:
        if pd.notna(row[col]):
            clinical_info["Grade"] = row[col]
            break

    vital_cols = [c for c in row.index if "vital" in c.lower()]
    for col in vital_cols:
        if pd.notna(row[col]):
            clinical_info["Vital Status"] = row[col]
            break

    diag_cols = [c for c in row.index if "primary_diagnosis" in c.lower()]
    for col in diag_cols:
        if pd.notna(row[col]):
            clinical_info["Diagnosis"] = row[col]
            break

    return clinical_info if clinical_info else None


def filter_by_thresholds(
    df: pd.DataFrame,
    proba_threshold: float = 0.95,
    margin_threshold: float = 0.20
) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()

    proba_col = None
    margin_col = None

    for col in ["top1_proba", "confidence", "proba", "probability"]:
        if col in df.columns:
            proba_col = col
            break

    for col in ["margin", "margin_score"]:
        if col in df.columns:
            margin_col = col
            break

    if proba_col is None or margin_col is None:
        return df

    mask = (df[proba_col] >= proba_threshold) & (df[margin_col] >= margin_threshold)
    return df[mask].copy()


def get_sample_prediction(df: pd.DataFrame, sample_id: str) -> Optional[Dict]:
    if df is None or sample_id not in df["sample_id"].values:
        return None

    row = df[df["sample_id"] == sample_id].iloc[0]

    prob_cols = detect_probability_columns(df)

    result = {
        "sample_id": sample_id,
        "predicted_subtype": row.get("predicted_subtype", "Unknown"),
        "confidence": row.get("top1_proba", row.get("confidence", 0)),
        "margin": row.get("margin", 0),
        "probabilities": {}
    }

    for subtype, col in prob_cols.items():
        result["probabilities"][subtype] = row[col]

    return result
