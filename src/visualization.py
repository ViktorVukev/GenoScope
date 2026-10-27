"""Dimensionality reduction and visualization for exploratory RNA-seq analysis.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# Ensure numba caching does not block in read-only / sandboxed site-packages environments
try:
    import numba.core.caching  # type: ignore
    numba.core.caching._SourceFileBackedLocatorMixin.ensure_cache_path = lambda self: None
except Exception:
    pass

import umap

# Categorical palette defined in AGENTS.md
CATEGORICAL_PALETTE: List[str] = [
    "#0284C7",  # Clinical Sky Blue
    "#0D9488",  # Genomic Teal
    "#D97706",  # Amber
    "#E11D48",  # Crimson Rose
    "#7C3AED",  # Violet
    "#2563EB",  # Royal Blue
]


def compute_pca_2d(
    features: Union[pd.DataFrame, np.ndarray],
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 2D Principal Component Analysis (PCA) projection.

    Parameters
    ----------
    features : Union[pd.DataFrame, np.ndarray]
        Numeric feature matrix (patients as rows, genes as columns).
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - 2D coordinates array of shape (n_samples, 2).
        - Explained variance ratio array for the 2 components.
    """
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(features)
    variance_ratio = pca.explained_variance_ratio_
    return coords, variance_ratio


def compute_umap_2d(
    features: Union[pd.DataFrame, np.ndarray],
    n_neighbors: Optional[int] = None,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Compute 2D Uniform Manifold Approximation and Projection (UMAP).

    Parameters
    ----------
    features : Union[pd.DataFrame, np.ndarray]
        Numeric feature matrix (patients as rows, genes as columns).
    n_neighbors : Optional[int], default=None
        Size of local neighborhood. If None or larger than sample size,
        dynamically bounded based on the number of samples.
    min_dist : float, default=0.1
        Minimum distance between points in low-dimensional space.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    np.ndarray
        2D coordinates array of shape (n_samples, 2).
    """
    n_samples = len(features)
    if n_neighbors is None or n_neighbors >= n_samples:
        effective_neighbors = max(2, min(15, n_samples - 1))
    else:
        effective_neighbors = max(2, n_neighbors)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    coords = reducer.fit_transform(features)
    return coords


def create_scatter_figure(
    coords: np.ndarray,
    labels: pd.Series,
    patient_ids: pd.Series,
    title: str,
    x_label: str,
    y_label: str,
) -> go.Figure:
    """Create an interactive 2D scatter plot using the project clinical palette.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (n_samples, 2) containing 2D projected coordinates.
    labels : pd.Series
        Cancer diagnosis category for each patient.
    patient_ids : pd.Series
        Identifier label for each patient.
    title : str
        Chart header title.
    x_label : str
        X-axis label.
    y_label : str
        Y-axis label.

    Returns
    -------
    go.Figure
        Configured Plotly figure.
    """
    df_plot = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "diagnosis": labels.values,
        "patient": patient_ids.values,
    })

    unique_classes = sorted(df_plot["diagnosis"].unique())
    fig = go.Figure()

    for idx, cancer_type in enumerate(unique_classes):
        class_subset = df_plot[df_plot["diagnosis"] == cancer_type]
        color = CATEGORICAL_PALETTE[idx % len(CATEGORICAL_PALETTE)]

        hover_template = (
            "<b>Patient:</b> %{customdata[0]}<br>"
            "<b>Diagnosis:</b> %{customdata[1]}<br>"
            f"<b>{x_label}:</b> %{{x:.2f}}<br>"
            f"<b>{y_label}:</b> %{{y:.2f}}<extra></extra>"
        )

        fig.add_trace(
            go.Scatter(
                x=class_subset["x"],
                y=class_subset["y"],
                mode="markers",
                name=str(cancer_type),
                marker=dict(
                    size=10,
                    color=color,
                    line=dict(width=1.0, color="#0F172A"),
                    opacity=0.88,
                ),
                customdata=np.stack(
                    (class_subset["patient"], class_subset["diagnosis"]),
                    axis=-1,
                ),
                hovertemplate=hover_template,
            )
        )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color="#0F172A", family="sans-serif"),
            x=0.0,
            y=0.96,
        ),
        xaxis=dict(
            title=dict(text=x_label, font=dict(color="#0F172A", family="sans-serif")),
            gridcolor="#E2E8F0",
            zeroline=False,
            showline=True,
            linecolor="#CBD5E1",
            tickfont=dict(color="#475569", family="sans-serif"),
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(color="#0F172A", family="sans-serif")),
            gridcolor="#E2E8F0",
            zeroline=False,
            showline=True,
            linecolor="#CBD5E1",
            tickfont=dict(color="#475569", family="sans-serif"),
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend=dict(
            title=dict(text="Diagnosis", font=dict(size=12, color="#0F172A")),
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(color="#0F172A", family="sans-serif"),
        ),
        margin=dict(l=40, r=40, t=50, b=70),
        height=450,
    )

    return fig


def plot_pca(
    features: Union[pd.DataFrame, np.ndarray],
    labels: pd.Series,
    patient_ids: pd.Series,
    random_state: int = 42,
) -> go.Figure:
    """Generate 2D PCA scatter plot with explained variance annotations.

    Parameters
    ----------
    features : Union[pd.DataFrame, np.ndarray]
        Numeric feature matrix.
    labels : pd.Series
        Cancer diagnosis category for each patient.
    patient_ids : pd.Series
        Identifier label for each patient.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    go.Figure
        Interactive Plotly figure for PCA.
    """
    coords, variance_ratio = compute_pca_2d(features, random_state=random_state)
    pc1_var = variance_ratio[0] * 100.0
    pc2_var = variance_ratio[1] * 100.0
    total_var = pc1_var + pc2_var

    title = f"2D PCA Projection (Total Explained Variance: {total_var:.1f}%)"
    x_label = f"PC1 ({pc1_var:.1f}% variance)"
    y_label = f"PC2 ({pc2_var:.1f}% variance)"

    return create_scatter_figure(
        coords=coords,
        labels=labels,
        patient_ids=patient_ids,
        title=title,
        x_label=x_label,
        y_label=y_label,
    )


def plot_umap(
    features: Union[pd.DataFrame, np.ndarray],
    labels: pd.Series,
    patient_ids: pd.Series,
    n_neighbors: Optional[int] = None,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> go.Figure:
    """Generate 2D UMAP scatter plot.

    Parameters
    ----------
    features : Union[pd.DataFrame, np.ndarray]
        Numeric feature matrix.
    labels : pd.Series
        Cancer diagnosis category for each patient.
    patient_ids : pd.Series
        Identifier label for each patient.
    n_neighbors : Optional[int], default=None
        UMAP neighborhood size.
    min_dist : float, default=0.1
        UMAP minimum distance.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    go.Figure
        Interactive Plotly figure for UMAP.
    """
    coords = compute_umap_2d(
        features,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    title = "2D UMAP Manifold Projection"
    x_label = "UMAP Dimension 1"
    y_label = "UMAP Dimension 2"

    return create_scatter_figure(
        coords=coords,
        labels=labels,
        patient_ids=patient_ids,
        title=title,
        x_label=x_label,
        y_label=y_label,
    )
