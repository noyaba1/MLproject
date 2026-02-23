from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go


SUBTYPE_COLORS = {
    "Basal": "#e41a1c",
    "Her2": "#984ea3",
    "LumA": "#4daf4a",
    "LumB": "#377eb8",
    "Normal": "#ff7f00",
}

SUBTYPES_ORDER = ["Basal", "Her2", "LumA", "LumB", "Normal"]


def create_probability_bar_chart(
    probabilities: Dict[str, float],
    predicted_class: str,
    title: str = "Class Probabilities"
) -> go.Figure:
    sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    subtypes = [item[0] for item in sorted_items]
    probs = [item[1] for item in sorted_items]

    colors = []
    for subtype in subtypes:
        if subtype == predicted_class:
            colors.append(SUBTYPE_COLORS.get(subtype, "#1f77b4"))
        else:
            colors.append("#d3d3d3")

    fig = go.Figure(go.Bar(
        x=probs,
        y=subtypes,
        orientation="h",
        marker_color=colors,
        text=[f"{p:.1%}" for p in probs],
        textposition="outside",
        textfont=dict(size=12, color="#212121"),
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#212121")),
        xaxis=dict(
            title=dict(text="Probability", font=dict(color="#424242")),
            range=[0, 1.15],
            tickformat=".0%",
            tickfont=dict(color="#424242"),
        ),
        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=subtypes[::-1],
            tickfont=dict(color="#212121"),
        ),
        height=250,
        margin=dict(l=80, r=20, t=40, b=40),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def create_subtype_distribution_chart(
    distribution: Dict[str, int],
    title: str = "Predicted Subtype Distribution"
) -> go.Figure:
    subtypes = list(distribution.keys())
    counts = list(distribution.values())
    colors = [SUBTYPE_COLORS.get(s, "#999999") for s in subtypes]

    fig = go.Figure(go.Pie(
        labels=subtypes,
        values=counts,
        hole=0.4,
        marker_colors=colors,
        textinfo="label+percent",
        textposition="outside",
        textfont=dict(size=11, color="#212121"),
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#212121")),
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(color="#212121"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def create_probability_histogram(
    df: pd.DataFrame,
    column: str = "top1_proba",
    title: str = "Confidence Distribution",
    threshold: Optional[float] = None
) -> go.Figure:
    if column not in df.columns:
        for alt in ["confidence", "proba", "probability"]:
            if alt in df.columns:
                column = alt
                break

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df[column],
        nbinsx=30,
        marker_color="#377eb8",
        opacity=0.75,
        name="Distribution",
    ))

    if threshold is not None:
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold: {threshold:.0%}",
            annotation_position="top right",
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#212121")),
        xaxis=dict(
            title=dict(text=column.replace("_", " ").title(), font=dict(color="#424242")),
            tickformat=".0%",
            tickfont=dict(color="#424242"),
        ),
        yaxis=dict(
            title=dict(text="Count", font=dict(color="#424242")),
            tickfont=dict(color="#424242"),
        ),
        height=250,
        margin=dict(l=60, r=20, t=50, b=50),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def create_gene_coefficient_chart(
    genes_df: pd.DataFrame,
    n_genes: int = 10,
    title: str = "Top Contributing Genes"
) -> go.Figure:
    if genes_df is None or genes_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No gene data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig.update_layout(height=200)
        return fig

    df = genes_df.head(n_genes).copy()

    colors = ["#4daf4a" if c > 0 else "#e41a1c" for c in df["coefficient"]]

    fig = go.Figure(go.Bar(
        x=df["coefficient"],
        y=df["gene"],
        orientation="h",
        marker_color=colors,
        text=[f"{c:.3f}" for c in df["coefficient"]],
        textposition="outside",
        textfont=dict(size=10, color="#212121"),
    ))

    max_abs = max(abs(df["coefficient"].min()), abs(df["coefficient"].max()))
    x_range = [-max_abs * 1.3, max_abs * 1.3] if df["coefficient"].min() < 0 else [0, max_abs * 1.3]

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#212121")),
        xaxis=dict(
            title=dict(text="Coefficient", font=dict(color="#424242")),
            range=x_range,
            tickfont=dict(color="#424242"),
        ),
        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=df["gene"].tolist()[::-1],
            tickfont=dict(color="#212121"),
        ),
        height=max(200, 30 * len(df) + 80),
        margin=dict(l=100, r=40, t=50, b=40),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def create_classification_report_table(report_df: pd.DataFrame) -> go.Figure:
    if report_df is None or report_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No classification report available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )
        return fig

    df = report_df.reset_index() if report_df.index.name or report_df.index[0] != 0 else report_df.copy()

    if df.columns[0] == "index" or df.columns[0] == "":
        df = df.rename(columns={df.columns[0]: "Class"})

    for col in df.columns[1:]:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")

    n_rows = len(df)
    fill_colors = []
    for i in range(n_rows):
        row_val = str(df.iloc[i, 0]).lower() if len(df.columns) > 0 else ""
        if "accuracy" in row_val or "avg" in row_val:
            fill_colors.append("#e3f2fd")
        else:
            fill_colors.append("#fafafa")

    fig = go.Figure(go.Table(
        header=dict(
            values=list(df.columns),
            fill_color="#1565c0",
            font=dict(color="white", size=12),
            align="center",
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color=[fill_colors] * len(df.columns),
            font=dict(color="#212121", size=11),
            align=["left"] + ["center"] * (len(df.columns) - 1),
            height=28,
        ),
    ))

    fig.update_layout(
        height=max(200, 35 * len(df) + 50),
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig
