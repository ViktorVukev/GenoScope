"""
src/evaluation.py
Handles model evaluation metrics and interactive Plotly visualizations for Step 4.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes overall classification accuracy.
    """
    accuracy = accuracy_score(y_true, y_pred)
    return {
        "accuracy": float(accuracy),
        "accuracy_pct": float(accuracy * 100.0)
    }


def plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    classes: List[str]
) -> go.Figure:
    """
    Creates an interactive annotated heatmap for the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    fig = px.imshow(
        cm,
        x=classes,
        y=classes,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted Diagnosis", y="True Diagnosis", color="Count"),
        title="Confusion Matrix"
    )

    fig.update_layout(
        width=550,
        height=500,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def plot_roc_curves(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    classes: List[str]
) -> go.Figure:
    """
    Plots multi-class One-vs-Rest (OvR) ROC curves with AUC scores.
    """
    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)
    n_classes = len(classes)

    # Binarize labels for multi-class ROC
    y_bin = label_binarize(y_true_arr, classes=classes)
    if n_classes == 2 and y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig = go.Figure()

    # Calculate ROC and AUC per class
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob_arr[:, i])
        roc_auc = auc(fpr, tpr)

        fig.add_trace(
            go.Scatter(
                x=fpr, 
                y=tpr, 
                mode="lines", 
                name=f"{class_name} (AUC = {roc_auc:.2f})",
                line=dict(width=2)
            )
        )

    # Reference diagonal line (random chance)
    fig.add_trace(
        go.Scatter(
            x=[0, 1], 
            y=[0, 1], 
            mode="lines", 
            name="Random Chance",
            line=dict(dash="dash", color="grey")
        )
    )

    fig.update_layout(
        title="Multi-Class ROC Curves (One-vs-Rest)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0.0, 1.05]),
        yaxis=dict(range=[0.0, 1.05]),
        width=600,
        height=500,
        legend=dict(x=0.45, y=0.15)
    )
    return fig


def plot_feature_importance(
    model, 
    gene_cols: List[str], 
    top_n: int = 10,
    model_name: str = "Model"
) -> Optional[go.Figure]:
    """
    Extracts top predictive genes using model weights or feature importances.
    Works for tree models (feature_importances_) and linear models (coef_).
    """
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = np.abs(model.coef_)
        importances = np.mean(coef, axis=0) if coef.ndim > 1 else coef

    if importances is None:
        return None

    # Sort genes by importance
    df_importance = pd.DataFrame({
        "gene": gene_cols,
        "importance": importances
    }).sort_values(by="importance", ascending=True)

    df_top = df_importance.tail(top_n)

    fig = go.Figure(
        go.Bar(
            x=df_top["importance"],
            y=df_top["gene"],
            orientation="h",
            marker=dict(color="#1f77b4")
        )
    )

    fig.update_layout(
        title=f"Top {len(df_top)} Biomarker Genes ({model_name})",
        xaxis_title="Importance Score",
        yaxis_title="Gene",
        height=400,
        width=600,
        margin=dict(l=60, r=40, t=60, b=40)
    )
    return fig