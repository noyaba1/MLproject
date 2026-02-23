"""
PAM50 Prediction Explorer Dashboard

A Streamlit dashboard for exploring PAM50 breast cancer subtype predictions.
This is a demonstration tool - NOT for clinical use.

Run with: streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.data_loader import (
    SUBTYPES,
    detect_probability_columns,
    filter_by_thresholds,
    get_clinical_for_sample,
    get_figure_path,
    get_sample_prediction,
    get_top_genes_for_class,
    load_classification_report,
    load_clinical_data,
    load_model_metrics,
    load_pseudolabels_high_confidence,
    load_pseudolabels_summary,
    load_pseudolabels_with_clinical,
    load_top_genes,
    load_unlabeled_predictions,
)
from src.dashboard.plots import (
    SUBTYPE_COLORS,
    create_classification_report_table,
    create_gene_coefficient_chart,
    create_probability_bar_chart,
    create_probability_histogram,
    create_subtype_distribution_chart,
)


# Page configuration
st.set_page_config(
    page_title="PAM50 Prediction Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling - with proper contrast for readability
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1565c0;
        margin-bottom: 0.5rem;
    }
    .disclaimer-box {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .disclaimer-box ul {
        margin-bottom: 0;
        color: #664d03 !important;
    }
    .disclaimer-box li {
        color: #664d03 !important;
    }
    .disclaimer-title {
        color: #664d03 !important;
        font-weight: bold;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    .prediction-box {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        border: 2px solid #90caf9;
    }
    .prediction-label {
        font-size: 1rem;
        color: #1565c0 !important;
        font-weight: 500;
    }
    .prediction-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #f5f5f5;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 2px solid #e0e0e0;
    }
    .metric-card-label {
        color: #424242 !important;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .metric-card-value {
        font-size: 1.8rem;
        font-weight: bold;
    }
    .exploratory-badge {
        background-color: #d32f2f;
        color: white !important;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .clinical-card {
        background-color: #fafafa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 2px solid #e0e0e0;
        margin-bottom: 0.5rem;
    }
    .clinical-label {
        color: #424242 !important;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .clinical-value {
        color: #1a1a1a !important;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .info-box {
        background-color: #e3f2fd;
        border: 2px solid #90caf9;
        border-radius: 8px;
        padding: 1rem;
        color: #0d47a1 !important;
    }
    .footer-text {
        text-align: center;
        color: #757575 !important;
        font-size: 0.85rem;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Data Loading (cached)
# ============================================================================

@st.cache_data
def load_all_data():
    """Load all data files with caching."""
    data = {}

    # Load predictions
    data["unlabeled"], data["unlabeled_status"] = load_unlabeled_predictions()
    data["pseudolabels"], data["pseudolabels_status"] = load_pseudolabels_high_confidence()
    data["pseudolabels_clinical"], data["pseudolabels_clinical_status"] = load_pseudolabels_with_clinical()

    # Load supporting data
    data["clinical"], data["clinical_status"] = load_clinical_data()
    data["top_genes"], data["top_genes_status"] = load_top_genes()
    data["metrics"], data["metrics_status"] = load_model_metrics()
    data["classification_report"], data["classification_report_status"] = load_classification_report()
    data["pseudolabels_summary"], data["summary_status"] = load_pseudolabels_summary()

    return data


# Load data
data = load_all_data()


# ============================================================================
# Sidebar
# ============================================================================

st.sidebar.title("PAM50 Explorer")
st.sidebar.markdown("---")

# Data source selector
st.sidebar.subheader("Data Source")

data_sources = []
if data["pseudolabels"] is not None:
    data_sources.append("High-confidence pseudo-labels (exploratory)")
if data["unlabeled"] is not None:
    data_sources.append("All unlabeled predictions")

if not data_sources:
    st.sidebar.error("No prediction data available!")
    st.stop()

selected_source = st.sidebar.selectbox(
    "Select data to explore:",
    options=data_sources,
    index=0,
)

# Determine which dataframe to use
if "High-confidence" in selected_source:
    # Prefer clinical-enriched version if available
    if data["pseudolabels_clinical"] is not None:
        current_df = data["pseudolabels_clinical"]
    else:
        current_df = data["pseudolabels"]
    is_exploratory = True
else:
    current_df = data["unlabeled"]
    is_exploratory = True  # All unlabeled are exploratory

# Threshold controls (for all unlabeled)
if "All unlabeled" in selected_source:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filter Thresholds")

    proba_threshold = st.sidebar.slider(
        "Min Probability",
        min_value=0.5,
        max_value=1.0,
        value=0.95,
        step=0.05,
        format="%.2f",
    )

    margin_threshold = st.sidebar.slider(
        "Min Margin",
        min_value=0.0,
        max_value=0.5,
        value=0.20,
        step=0.05,
        format="%.2f",
    )

    if st.sidebar.button("Apply Filter"):
        current_df = filter_by_thresholds(current_df, proba_threshold, margin_threshold)
        st.sidebar.success(f"Filtered to {len(current_df)} samples")

st.sidebar.markdown("---")

# Sample selector
st.sidebar.subheader("Sample Selection")

if current_df is not None and len(current_df) > 0:
    sample_ids = current_df["sample_id"].tolist()

    selected_sample = st.sidebar.selectbox(
        "Select sample:",
        options=sample_ids,
        index=0,
        help="Search by TCGA sample ID",
    )

    # Show sample count
    st.sidebar.caption(f"Total samples: {len(sample_ids)}")
else:
    st.sidebar.warning("No samples available")
    selected_sample = None

# Data status
st.sidebar.markdown("---")
st.sidebar.subheader("Data Status")

status_items = [
    ("Predictions", data["unlabeled_status"]),
    ("Pseudo-labels", data["pseudolabels_status"]),
    ("Clinical", data["clinical_status"]),
    ("Gene coefficients", data["top_genes_status"]),
]

for name, status in status_items:
    if "Loaded" in status or "loaded" in status.lower():
        st.sidebar.success(f"✓ {name}")
    else:
        st.sidebar.warning(f"✗ {name}")


# ============================================================================
# Main Content
# ============================================================================

# Header
st.markdown('<div class="main-header">PAM50 Prediction Explorer</div>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer-box">
    <div class="disclaimer-title">Important Disclaimer</div>
    <ul>
        <li><strong>NOT FOR CLINICAL USE</strong> - This is a demonstration tool only</li>
        <li><strong>Pseudo-labels are NOT ground truth</strong> - They are model predictions, not validated subtypes</li>
        <li>Dashboard is for interpretation and exploration purposes only</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Main layout
if selected_sample:
    # Get prediction for selected sample
    prediction = get_sample_prediction(current_df, selected_sample)

    if prediction:
        # Two-column layout for prediction and probabilities
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Prediction")

            # Exploratory badge
            if is_exploratory:
                st.markdown('<span class="exploratory-badge">EXPLORATORY (no ground truth)</span>',
                           unsafe_allow_html=True)

            # Predicted subtype
            subtype = prediction["predicted_subtype"]
            color = SUBTYPE_COLORS.get(subtype, "#1f77b4")

            st.markdown(f"""
            <div class="prediction-box">
                <div class="prediction-label">Predicted Subtype</div>
                <div class="prediction-value" style="color: {color};">{subtype}</div>
                <div class="prediction-label">Sample: <strong>{selected_sample}</strong></div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence metrics
            st.markdown("---")
            conf_col1, conf_col2 = st.columns(2)

            with conf_col1:
                confidence = prediction["confidence"]
                conf_color = "#2e7d32" if confidence >= 0.95 else "#e65100"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-card-label">Confidence</div>
                    <div class="metric-card-value" style="color: {conf_color};">
                        {confidence:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with conf_col2:
                margin = prediction["margin"]
                margin_color = "#2e7d32" if margin >= 0.20 else "#e65100"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-card-label">Margin</div>
                    <div class="metric-card-value" style="color: {margin_color};">
                        {margin:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.subheader("Class Probabilities")

            # Probability bar chart
            if prediction["probabilities"]:
                fig = create_probability_bar_chart(
                    prediction["probabilities"],
                    prediction["predicted_subtype"],
                    title=""
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Probability data not available")

        # Gene explanation section
        st.markdown("---")
        st.subheader("Model Explanation: Top Contributing Genes")

        exp_col1, exp_col2 = st.columns([2, 1])

        with exp_col1:
            if data["top_genes"] is not None:
                # Get top genes for predicted class
                top_genes = get_top_genes_for_class(
                    data["top_genes"],
                    prediction["predicted_subtype"],
                    n=10,
                    direction="positive"
                )

                if not top_genes.empty:
                    fig = create_gene_coefficient_chart(
                        top_genes,
                        n_genes=10,
                        title=f"Top Genes for {prediction['predicted_subtype']} Subtype"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No gene coefficients found for {prediction['predicted_subtype']}")
            else:
                st.warning("Gene coefficient data not available")

        with exp_col2:
            st.info("""
            **About Gene Coefficients**

            These are ElasticNet logistic regression coefficients learned from
            the labeled training data.

            - **Positive coefficients**: Higher expression associated with this subtype
            - **Negative coefficients**: Lower expression associated with this subtype

            This is a **global model explanation** (coefficients apply to all samples),
            not a sample-specific explanation like SHAP values.
            """)

        # Clinical context section
        st.markdown("---")
        st.subheader("Clinical Context")

        # Try to get clinical data
        clinical_info = None

        # Check if we have enriched data with clinical columns embedded in current_df
        if data["pseudolabels_clinical"] is not None:
            row = current_df[current_df["sample_id"] == selected_sample]
            if not row.empty:
                row = row.iloc[0]
                clinical_info = {}

                # Extract clinical columns from enriched data
                clinical_cols = {
                    "age_at_index.demographic": "Age",
                    "gender.demographic": "Gender",
                    "ajcc_pathologic_stage.diagnoses": "Stage",
                    "tumor_grade.diagnoses": "Grade",
                    "vital_status.demographic": "Vital Status",
                    "primary_diagnosis.diagnoses": "Diagnosis",
                }

                for col, label in clinical_cols.items():
                    if col in row.index and pd.notna(row[col]) and str(row[col]) != "Not Reported":
                        clinical_info[label] = row[col]

        # Fall back to clinical data file
        if not clinical_info and data["clinical"] is not None:
            clinical_info = get_clinical_for_sample(data["clinical"], selected_sample)

        if clinical_info:
            clin_cols = st.columns(len(clinical_info))
            for i, (label, value) in enumerate(clinical_info.items()):
                with clin_cols[i]:
                    st.markdown(f"""
                    <div class="clinical-card">
                        <div class="clinical-label">{label}</div>
                        <div class="clinical-value">{value}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No clinical data found for this sample/patient.")

else:
    st.info("Select a sample from the sidebar to view predictions.")

# ============================================================================
# Project-level Results (Tabs)
# ============================================================================

st.markdown("---")
st.header("Project-Level Results")

tab1, tab2, tab3 = st.tabs(["Model Performance", "High-Confidence Summary", "Data Notes"])

with tab1:
    st.subheader("Model Performance (5-Fold Stratified CV)")

    perf_col1, perf_col2 = st.columns([1, 1])

    with perf_col1:
        # Key metrics
        if data["metrics"] is not None:
            metrics = data["metrics"].iloc[0]

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                macro_f1 = metrics.get("f1_macro_mean", 0)
                st.metric("Macro-F1 (CV)", f"{macro_f1:.3f}")
            with m_col2:
                accuracy = metrics.get("accuracy_mean", 0)
                st.metric("Accuracy (CV)", f"{accuracy:.3f}")
            with m_col3:
                shuffled = metrics.get("shuffled_f1", 0)
                st.metric("Shuffled-label F1", f"{shuffled:.3f}",
                          help="Permutation check: F1 with randomly shuffled labels. Should be near random chance (~0.25). Large gap vs CV F1 confirms no data leakage.")
            with m_col4:
                gap = metrics.get("overfit_gap", 0)
                st.metric("Overfit Gap", f"{gap:.3f}",
                          help="Train F1 minus CV F1. Non-zero gap expected in p>>n settings (298:1 here).")
        else:
            st.warning("Model metrics not available")

        # Classification report
        st.markdown("#### Per-Class Metrics")
        if data["classification_report"] is not None:
            fig = create_classification_report_table(data["classification_report"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Classification report not available")

    with perf_col2:
        # Confusion matrix image
        st.markdown("#### Confusion Matrix")
        cm_path = get_figure_path("confusion_matrix.png")
        if cm_path:
            st.image(str(cm_path), use_container_width=True)
        else:
            st.info("Confusion matrix image not available")

with tab2:
    st.subheader("High-Confidence Pseudo-Labels Summary")

    if data["pseudolabels_summary"]:
        summary = data["pseudolabels_summary"]
        stats = summary.get("statistics", {})

        # Warning banner
        st.warning(summary.get("warning", "Pseudo-labels are not ground truth!"))

        # Key stats
        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)

        with sum_col1:
            st.metric("Total Unlabeled", f"{stats.get('n_total_unlabeled', 0):,}")
        with sum_col2:
            st.metric("Selected", f"{stats.get('n_selected', 0):,}")
        with sum_col3:
            rate = stats.get("selection_rate", 0)
            st.metric("Selection Rate", f"{rate:.1%}")
        with sum_col4:
            st.metric("Rejected", f"{stats.get('n_rejected', 0):,}")

        # Criteria
        st.markdown("#### Selection Criteria")
        criteria = summary.get("selection_criteria", {})
        st.code(f"""
Probability threshold: >= {criteria.get('proba_threshold', 0.95)}
Margin threshold:      >= {criteria.get('margin_threshold', 0.20)}
        """)

        # Distribution charts
        dist_col1, dist_col2 = st.columns(2)

        with dist_col1:
            distribution = stats.get("selected_distribution", {})
            if distribution:
                fig = create_subtype_distribution_chart(
                    distribution,
                    title="Selected Samples by Subtype"
                )
                st.plotly_chart(fig, use_container_width=True)

        with dist_col2:
            if data["unlabeled"] is not None:
                fig = create_probability_histogram(
                    data["unlabeled"],
                    column="top1_proba",
                    title="Confidence Distribution (All Unlabeled)",
                    threshold=0.95
                )
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Pseudo-labeling summary not available. Run Notebook 05 (05_unlabeled_inference.ipynb) to generate predictions.")

with tab3:
    st.subheader("Data Notes")

    # Load data truth table if available for dynamic counts
    import json as _json
    _truth_path = PROJECT_ROOT / "reports" / "tables" / "data_truth_table.json"
    if _truth_path.exists():
        with open(_truth_path, encoding="utf-8") as _f:
            _truth = _json.load(_f)
        _raw_rows   = _truth.get("raw_label_rows", "?")
        _unique     = _truth.get("unique_after_dedup", "?")
        _labeled    = _truth.get("after_unknown_exclusion", "?")
        _training   = _truth.get("used_in_training", "?")
        _unlabeled  = _truth.get("unlabeled", "?")
        _expr_samples = _truth.get("expression_samples", "?")
    else:
        _raw_rows, _unique, _labeled, _training, _unlabeled, _expr_samples = "?", "?", "?", "?", "?", "?"

    # Format helper: use integer comma-format when value is a number, else plain string
    def _fmt(v):
        return f"{v:,}" if isinstance(v, int) else str(v)

    st.markdown(f"""
    ### Labeled Data Summary

    | Metric | Value |
    |--------|-------|
    | Expression samples (HiSeqV2.csv) | {_fmt(_expr_samples)} |
    | PAM50 raw label rows | {_raw_rows} |
    | Unique samples after deduplication | {_unique} |
    | **Labeled samples** (Unknown removed) | **{_labeled}** |
    | Normal excluded (< 2 samples) | 1 |
    | **Supervised training samples** | **{_training}** |
    | Unlabeled samples (expression only) | {_fmt(_unlabeled)} |

    ### Data Processing

    The PAM50 labels file ({_raw_rows} rows) undergoes:
    1. **TCGA barcode normalization**: All IDs truncated to 15 chars for matching
    2. **Deduplication**: Duplicate sample IDs removed (keep first), {_raw_rows} → {_unique} unique
    3. **Exclusion**: Samples with "Unknown" subtype are removed (4 samples)
    4. **Rare-class exclusion**: Normal (1 sample) excluded from training

    After processing, **{_training} samples** are used for supervised learning.
    Counts are read at runtime from `reports/tables/data_truth_table.json`.

    ### Label Distribution

    The label distribution is computed at runtime from the actual data.
    Run Notebook 01 (`01_data_audit.ipynb`) to recompute the current distribution.

    **Note**: Class imbalance is handled via `class_weight='balanced'` and
    stratified cross-validation.

    ### Pseudo-Labeling Criteria

    High-confidence pseudo-labels are selected using:
    - **Probability threshold**: Top-1 probability >= 0.95
    - **Margin threshold**: (Top-1 - Top-2) probability >= 0.20

    These thresholds ensure only highly confident predictions are included,
    but they are still **model predictions, not ground truth**.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer-text">
    PAM50 Prediction Explorer | Demonstration Tool | Not for Clinical Use
</div>
""", unsafe_allow_html=True)
