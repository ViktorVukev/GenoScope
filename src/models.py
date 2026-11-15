"""Machine learning and deep learning models for RNA-seq cancer classification.
Implements Step 3: Neural Network (MLP), Random Forest, Logistic Regression,
and Single-Patient Clinical Inference.
"""

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelResult:
    """Standardized container for a trained model and its test-set predictions.

    Attributes
    ----------
    model_key : str
        Internal identifier ('mlp', 'rf', 'lr').
    model_name : str
        Human-readable display name.
    estimator : BaseEstimator
        Fitted Scikit-learn model or Pipeline.
    y_pred : np.ndarray
        1D array of predicted cancer diagnoses on X_test, shape (n_test,).
    y_prob : np.ndarray
        2D array of predicted class probabilities on X_test, shape (n_test, n_classes).
    classes : List[str]
        Ordered list of class labels corresponding to columns of y_prob.
    feature_importances : np.ndarray
        1D array of non-negative importance scores per gene, shape (n_genes,).
    training_time_sec : float
        Wall-clock time taken to fit the model in seconds.
    """

    model_key: str
    model_name: str
    estimator: BaseEstimator
    y_pred: np.ndarray
    y_prob: np.ndarray
    classes: List[str]
    feature_importances: np.ndarray
    training_time_sec: float


def build_mlp_model(
    hidden_layer_sizes: Tuple[int, ...] = (128, 64),
    max_iter: int = 300,
    learning_rate_init: float = 0.001,
    random_state: int = 42,
) -> Pipeline:
    """Build a standardized Multi-Layer Perceptron (MLP) Neural Network pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    alpha=0.001,
                    learning_rate_init=learning_rate_init,
                    max_iter=max_iter,
                    early_stopping=True,
                    n_iter_no_change=15,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_random_forest_model(
    n_estimators: int = 200,
    max_depth: Union[int, None] = None,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Build a Random Forest classifier baseline."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )


def build_logistic_regression_model(
    c_param: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> Pipeline:
    """Build a standardized multinomial Logistic Regression baseline pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_param,
                    max_iter=max_iter,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def extract_feature_importances(model: BaseEstimator) -> np.ndarray:
    """Extract a 1D array of gene importance scores from a fitted model or Pipeline."""
    core_model = (
        model.named_steps["classifier"]
        if isinstance(model, Pipeline) and "classifier" in model.named_steps
        else model
    )

    if hasattr(core_model, "feature_importances_"):
        return np.asarray(core_model.feature_importances_, dtype=float)

    if hasattr(core_model, "coef_"):
        coef = np.abs(np.asarray(core_model.coef_, dtype=float))
        return np.mean(coef, axis=0) if coef.ndim > 1 else coef

    if hasattr(core_model, "coefs_") and len(core_model.coefs_) > 0:
        first_layer_weights = np.abs(np.asarray(core_model.coefs_[0], dtype=float))
        return np.mean(first_layer_weights, axis=1)

    raise ValueError(
        f"Cannot extract feature importances from model of type {type(core_model).__name__}."
    )


def train_and_predict_model(
    estimator: BaseEstimator,
    model_key: str,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    classes: List[str],
) -> ModelResult:
    """Fit a single model on (X_train, y_train) and generate blind predictions on X_test."""
    start_time = time.perf_counter()
    estimator.fit(X_train, y_train)
    elapsed = time.perf_counter() - start_time

    y_pred = np.asarray(estimator.predict(X_test))
    raw_prob = np.asarray(estimator.predict_proba(X_test), dtype=float)

    model_classes = list(estimator.classes_)
    if model_classes != list(classes):
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(model_classes)}
        aligned_prob = np.zeros((len(X_test), len(classes)), dtype=float)
        for target_idx, cls_name in enumerate(classes):
            if cls_name in class_to_idx:
                aligned_prob[:, target_idx] = raw_prob[:, class_to_idx[cls_name]]
        y_prob = aligned_prob
    else:
        y_prob = raw_prob

    importances = extract_feature_importances(estimator)

    return ModelResult(
        model_key=model_key,
        model_name=model_name,
        estimator=estimator,
        y_pred=y_pred,
        y_prob=y_prob,
        classes=list(classes),
        feature_importances=importances,
        training_time_sec=elapsed,
    )


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    classes: List[str],
    mlp_hidden_layers: Tuple[int, ...] = (128, 64),
    mlp_max_iter: int = 300,
    rf_n_estimators: int = 200,
    lr_c: float = 1.0,
    random_state: int = 42,
) -> Dict[str, ModelResult]:
    """Train all three benchmark models on the 80% split and predict on the 20% split."""
    mlp_estimator = build_mlp_model(
        hidden_layer_sizes=mlp_hidden_layers,
        max_iter=mlp_max_iter,
        random_state=random_state,
    )
    if len(X_train) < len(classes) * 10:
        mlp_estimator.named_steps["classifier"].early_stopping = False

    rf_estimator = build_random_forest_model(
        n_estimators=rf_n_estimators,
        random_state=random_state,
    )
    lr_estimator = build_logistic_regression_model(
        c_param=lr_c,
        random_state=random_state,
    )

    specifications = [
        ("mlp", "Neural Network (MLP)", mlp_estimator),
        ("rf", "Random Forest", rf_estimator),
        ("lr", "Logistic Regression", lr_estimator),
    ]

    results: Dict[str, ModelResult] = {}
    for key, name, estimator in specifications:
        results[key] = train_and_predict_model(
            estimator=estimator,
            model_key=key,
            model_name=name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            classes=classes,
        )

    return results


def predict_single_patient_df(
    patient_df: pd.DataFrame,
    model_results: Dict[str, ModelResult],
    gene_cols: List[str],
    patient_col: str,
    target_col: str,
) -> Dict[str, Any]:
    """Validate an uploaded single-patient DataFrame and predict cancer diagnosis across all models.

    Supports uploaded CSVs either WITH the ground-truth target column (for validation)
    or WITHOUT the target column (blind clinical inference).

    Parameters
    ----------
    patient_df : pd.DataFrame
        Uploaded DataFrame containing 1 or more patient rows.
    model_results : Dict[str, ModelResult]
        Trained model dictionary returned by train_all_models().
    gene_cols : List[str]
        List of required gene feature column names from training.
    patient_col : str
        Expected patient identifier column name.
    target_col : str
        Expected target diagnosis column name (optional in uploaded file).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'patient_id': str identifier of the first row
        - 'true_diagnosis': Optional[str] ground-truth label if present in CSV, else None
        - 'predictions': Dict[str, Dict[str, Any]] per-model predicted class, confidence,
          and full class probability breakdown.
    """
    if patient_df is None or patient_df.empty:
        raise ValueError("Uploaded patient file is empty.")

    missing_genes = [g for g in gene_cols if g not in patient_df.columns]
    if missing_genes:
        preview = ", ".join(missing_genes[:5])
        raise ValueError(
            f"Uploaded file is missing {len(missing_genes)} required gene columns "
            f"(e.g., {preview}). Ensure the file matches the training dataset genes."
        )

    # Extract first patient row
    row = patient_df.iloc[[0]]

    if patient_col in row.columns:
        patient_id = str(row[patient_col].iloc[0])
    else:
        patient_id = str(row.iloc[0, 0])

    true_diagnosis: Optional[str] = None
    if target_col in row.columns:
        val = row[target_col].iloc[0]
        if pd.notna(val) and str(val).strip() != "":
            true_diagnosis = str(val).strip()

    X_single = row[gene_cols].apply(pd.to_numeric, errors="coerce")
    if X_single.isnull().any().any():
        raise ValueError("Uploaded patient record contains non-numeric or missing gene values.")

    predictions: Dict[str, Dict[str, Any]] = {}
    for key, res in model_results.items():
        pred_label = str(res.estimator.predict(X_single)[0])
        raw_prob = np.asarray(res.estimator.predict_proba(X_single)[0], dtype=float)

        model_classes = list(res.estimator.classes_)
        prob_map = {cls_name: 0.0 for cls_name in res.classes}
        for cls_idx, cls_name in enumerate(model_classes):
            prob_map[str(cls_name)] = float(raw_prob[cls_idx])

        confidence = prob_map.get(pred_label, float(np.max(raw_prob)))

        predictions[key] = {
            "model_name": res.model_name,
            "predicted_class": pred_label,
            "confidence_pct": confidence * 100.0,
            "probabilities": prob_map,
            "is_correct": (pred_label == true_diagnosis) if true_diagnosis is not None else None,
        }

    return {
        "patient_id": patient_id,
        "true_diagnosis": true_diagnosis,
        "predictions": predictions,
    }