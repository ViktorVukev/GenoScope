"""
src/evaluation.py
Handles model evaluation metrics and interactive Plotly visualizations for Step 4.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

from src.models import extract_feature_importances
from src.visualization import CATEGORICAL_PALETTE

FEATURE_IMPORTANCE_COLOR: str = "#0D9488"  

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
        Plotly heatmap figure using the 'Blues' colormap. Each cell shows the
        patient count and the row percentage (share of that true diagnosis),
        and colour encodes the row percentage so classes of different sizes
        are comparable.
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    row_totals = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_pct = np.where(row_totals > 0, cm / row_totals * 100.0, 0.0)

    display_labels = [_wrap_label(c) for c in classes]
    cell_text = [
        [f"{cm[i, j]}<br>({row_pct[i, j]:.0f}%)" for j in range(len(classes))]
        for i in range(len(classes))
    ]
    hover_counts = np.dstack([cm, np.broadcast_to(row_totals, cm.shape)])

    fig = go.Figure(
        go.Heatmap(
            z=row_pct,
            x=display_labels,
            y=display_labels,
            text=cell_text,
            texttemplate="%{text}",
            textfont=dict(size=14),
            customdata=hover_counts,
            colorscale="Blues",
            zmin=0,
            zmax=100,
            showscale=False,
            xgap=2,
            ygap=2,
            hovertemplate=(
                "<b>True:</b> %{y}<br><b>Predicted:</b> %{x}<br>"
                "<b>Patients:</b> %{customdata[0]} of %{customdata[1]} "
                "(%{z:.0f}%)<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"Confusion Matrix ({model_name})",
        xaxis=dict(title="Predicted Diagnosis", tickangle=0, side="bottom"),
        yaxis=dict(title="True Diagnosis", autorange="reversed"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A", family="sans-serif"),
        height=460,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def _wrap_label(label: str, max_line: int = 12) -> str:
    """Break a long class name onto two lines so axis labels stay horizontal."""
    text = str(label)
    if len(text) <= max_line or " " not in text:
        return text
    words = text.split(" ")
    best_split, best_diff = 1, len(text)
    for k in range(1, len(words)):
        left = " ".join(words[:k])
        right = " ".join(words[k:])
        diff = abs(len(left) - len(right))
        if diff < best_diff:
            best_split, best_diff = k, diff
    return " ".join(words[:best_split]) + "<br>" + " ".join(words[best_split:])


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

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Chance (AUC = 0.500)",
            line=dict(dash="dash", color="#94A3B8", width=1.5),
            hoverinfo="skip",
            legendrank=2000,
        )
    )

    # When several classes are perfectly separated their curves lie exactly on
    # top of each other. Varying width and dash keeps every class visible.
    dash_styles = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
    n_curves = 0
    aucs: List[float] = []
    skipped: List[str] = []

    for i, class_name in enumerate(classes):
        positives = int(y_bin[:, i].sum())
        if positives == 0 or positives == len(y_bin):
            skipped.append(str(class_name))
            continue

        color = CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob_arr[:, i])
        roc_auc = float(auc(fpr, tpr))
        aucs.append(roc_auc)

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{class_name} (AUC = {roc_auc:.3f})",
                line=dict(
                    width=max(2.0, 6.0 - 1.5 * n_curves),
                    color=color,
                    dash=dash_styles[n_curves % len(dash_styles)],
                ),
                hovertemplate=(
                    f"<b>{class_name}</b><br>FPR: %{{x:.2f}}<br>TPR: %{{y:.2f}}<extra></extra>"
                ),
            )
        )
        n_curves += 1

    title = f"One-vs-Rest ROC Curves ({model_name})"
    if aucs:
        title += f"<br><sup>Macro-average AUC = {np.mean(aucs):.3f}</sup>"
    if skipped:
        title += (
            f"<br><sup>Not in test set (no curve): {', '.join(skipped)}</sup>"
        )

    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Sensitivity)",
        xaxis=dict(range=[-0.03, 1.03], gridcolor="#E2E8F0", zeroline=False, dtick=0.2),
        yaxis=dict(range=[-0.03, 1.05], gridcolor="#E2E8F0", zeroline=False, dtick=0.2),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A", family="sans-serif"),
        height=460,
        margin=dict(l=40, r=40, t=90, b=40),
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