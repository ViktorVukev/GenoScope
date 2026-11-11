"""
src/evaluation.py
Handles model evaluation metrics and interactive Plotly visualizations for Step 4.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

from src.models import extract_feature_importances
from src.visualization import CATEGORICAL_PALETTE

FEATURE_IMPORTANCE_COLOR: str = "#0D9488"  # Genomic Teal from AGENTS.md


def calculate_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> Dict[str, Any]:
    """Compute overall classification accuracy, patient counts, and formatted string.

    Parameters
    ----------
    y_true : Union[pd.Series, np.ndarray]
        Ground-truth diagnosis labels from the 20% test set.
    y_pred : Union[pd.Series, np.ndarray]
        Model-predicted diagnosis labels on the 20% test set.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'accuracy': float in [0.0, 1.0]
        - 'accuracy_pct': float in [0.0, 100.0]
        - 'correct_count': int number of correctly classified test patients
        - 'total_count': int total number of test patients
        - 'formatted_accuracy': str formatted as 'X / Y patients (Z.Z% accuracy)'
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    total_count = int(len(y_true_arr))
    correct_count = int(np.sum(y_true_arr == y_pred_arr))
    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
    accuracy_pct = float(accuracy * 100.0)

    return {
        "accuracy": accuracy,
        "accuracy_pct": accuracy_pct,
        "correct_count": correct_count,
        "total_count": total_count,
        "formatted_accuracy": (
            f"{correct_count} / {total_count} patients ({accuracy_pct:.1f}% accuracy)"
        ),
    }


def plot_confusion_matrix(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    classes: List[str],
    model_name: str = "Model",
) -> go.Figure:
    """Create an interactive annotated heatmap for the confusion matrix.

    Parameters
    ----------
    y_true : Union[pd.Series, np.ndarray]
        True test diagnosis labels.
    y_pred : Union[pd.Series, np.ndarray]
        Predicted test diagnosis labels.
    classes : List[str]
        Ordered list of diagnosis classes.
    model_name : str, default='Model'
        Name of the model to display in the title.

    Returns
    -------
    go.Figure
        Plotly heatmap figure using the 'Blues' colormap.
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    fig = px.imshow(
        cm,
        x=classes,
        y=classes,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted Diagnosis", y="True Diagnosis", color="Patients"),
        title=f"Confusion Matrix ({model_name})",
    )

    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A", family="sans-serif"),
        height=460,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_roc_curves(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: np.ndarray,
    classes: List[str],
    model_name: str = "Model",
) -> go.Figure:
    """Plot multi-class One-vs-Rest (OvR) ROC curves with AUC scores.

    Parameters
    ----------
    y_true : Union[pd.Series, np.ndarray]
        True test diagnosis labels.
    y_prob : np.ndarray
        2D array of predicted probabilities of shape (n_samples, n_classes).
    classes : List[str]
        Ordered list of diagnosis classes.
    model_name : str, default='Model'
        Name of the model to display in the title.

    Returns
    -------
    go.Figure
        Interactive Plotly ROC figure.
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    n_classes = len(classes)

    y_bin = label_binarize(y_true_arr, classes=classes)
    if n_classes == 2 and y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig = go.Figure()

    for i, class_name in enumerate(classes):
        color = CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob_arr[:, i])
        roc_auc = auc(fpr, tpr)

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{class_name} (AUC = {roc_auc:.3f})",
                line=dict(width=2.5, color=color),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Chance (AUC = 0.500)",
            line=dict(dash="dash", color="#94A3B8", width=1.5),
        )
    )

    fig.update_layout(
        title=f"One-vs-Rest ROC Curves ({model_name})",
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Sensitivity)",
        xaxis=dict(range=[0.0, 1.02], gridcolor="#E2E8F0"),
        yaxis=dict(range=[0.0, 1.05], gridcolor="#E2E8F0"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A", family="sans-serif"),
        height=460,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            x=0.98,
            y=0.05,
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="#E2E8F0",
            borderwidth=1,
        ),
    )
    return fig


def plot_feature_importance(
    model: Any,
    gene_cols: List[str],
    top_n: int = 15,
    model_name: str = "Model",
) -> Optional[go.Figure]:
    """Extract and plot top predictive biomarker genes for any supported model.

    Works for Random Forest (feature_importances_), Logistic Regression (coef_),
    and Neural Network MLP (coefs_[0] input-layer weights).

    Parameters
    ----------
    model : Any
        Fitted Scikit-learn estimator, Pipeline, or 1D importance array.
    gene_cols : List[str]
        List of gene feature names.
    top_n : int, default=15
        Number of top genes to display.
    model_name : str, default='Model'
        Name of the model for chart title.

    Returns
    -------
    Optional[go.Figure]
        Horizontal bar chart of top contributing genes.
    """
    try:
        if isinstance(model, np.ndarray):
            importances = model
        else:
            importances = extract_feature_importances(model)
    except Exception:
        return None

    df_importance = pd.DataFrame(
        {"gene": gene_cols, "importance": importances}
    ).sort_values(by="importance", ascending=True)

    df_top = df_importance.tail(min(top_n, len(df_importance)))

    fig = go.Figure(
        go.Bar(
            x=df_top["importance"],
            y=df_top["gene"],
            orientation="h",
            marker=dict(color=FEATURE_IMPORTANCE_COLOR),
            hovertemplate="<b>Gene:</b> %{y}<br><b>Importance:</b> %{x:.5f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Top {len(df_top)} Biomarker Genes ({model_name})",
        xaxis_title="Relative Importance Score",
        yaxis_title="Gene Symbol / ID",
        xaxis=dict(gridcolor="#E2E8F0"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A", family="sans-serif"),
        height=460,
        margin=dict(l=80, r=40, t=60, b=40),
    )
    return fig