"""Machine learning and deep learning models for RNA-seq cancer classification.
Implements Step 3: Neural Network (MLP), Random Forest, and Logistic Regression.
"""

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Tuple, Union
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
    """Build a standardized Multi-Layer Perceptron (MLP) Neural Network pipeline.

    Parameters
    ----------
    hidden_layer_sizes : Tuple[int, ...], default=(128, 64)
        Number of neurons in each hidden layer.
    max_iter : int, default=300
        Maximum number of training epochs.
    learning_rate_init : float, default=0.001
        Initial learning rate for the Adam optimizer.
    random_state : int, default=42
        Seed for reproducible weight initialization.

    Returns
    -------
    Pipeline
        Scikit-learn Pipeline with StandardScaler and MLPClassifier.
    """
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
    """Build a Random Forest classifier baseline.

    Parameters
    ----------
    n_estimators : int, default=200
        Number of decision trees in the forest.
    max_depth : Union[int, None], default=None
        Maximum depth of each tree. None expands until leaves are pure.
    random_state : int, default=42
        Seed for reproducible bootstrapping and feature subsampling.

    Returns
    -------
    RandomForestClassifier
        Configured Random Forest estimator.
    """
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
    """Build a standardized multinomial Logistic Regression baseline pipeline.

    Parameters
    ----------
    c_param : float, default=1.0
        Inverse of L2 regularization strength.
    max_iter : int, default=1000
        Maximum iterations for solver convergence.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    Pipeline
        Scikit-learn Pipeline with StandardScaler and LogisticRegression.
    """
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
    """Extract a 1D array of gene importance scores from a fitted model or Pipeline.

    Supports:
    - Tree ensembles via `feature_importances_` (Gini importance).
    - Linear models via `coef_` (mean absolute coefficient across classes).
    - MLP neural networks via `coefs_[0]` (mean absolute input-layer weight).

    Parameters
    ----------
    model : BaseEstimator
        Fitted Scikit-learn estimator or Pipeline.

    Returns
    -------
    np.ndarray
        1D NumPy array of importance scores of shape (n_genes,).

    Raises
    ------
    ValueError
        If the model does not expose recognizable feature importance attributes.
    """
    core_model = (
        model.named_steps["classifier"]
        if isinstance(model, Pipeline) and "classifier" in model.named_steps
        else model
    )

    # 1. Random Forest / Tree-based models
    if hasattr(core_model, "feature_importances_"):
        return np.asarray(core_model.feature_importances_, dtype=float)

    # 2. Logistic Regression / Linear models
    if hasattr(core_model, "coef_"):
        coef = np.abs(np.asarray(core_model.coef_, dtype=float))
        return np.mean(coef, axis=0) if coef.ndim > 1 else coef

    # 3. Multi-Layer Perceptron (MLPClassifier) input-layer weights
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
    """Fit a single model on (X_train, y_train) and generate blind predictions on X_test.

    Parameters
    ----------
    estimator : BaseEstimator
        Unfitted Scikit-learn model or Pipeline.
    model_key : str
        Short identifier key ('mlp', 'rf', 'lr').
    model_name : str
        Display name for UI and charts.
    X_train : pd.DataFrame
        Training gene expression matrix (80% split).
    y_train : pd.Series
        Training cancer diagnosis labels (80% split).
    X_test : pd.DataFrame
        Testing gene expression matrix (20% split, labels hidden).
    classes : List[str]
        Sorted list of distinct diagnosis classes.

    Returns
    -------
    ModelResult
        Structured result containing the fitted estimator, predictions, probabilities,
        and gene importance scores.
    """
    start_time = time.perf_counter()
    estimator.fit(X_train, y_train)
    elapsed = time.perf_counter() - start_time

    y_pred = np.asarray(estimator.predict(X_test))
    raw_prob = np.asarray(estimator.predict_proba(X_test), dtype=float)

    # Ensure probability columns strictly match the order of `classes`
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
    """Train all three benchmark models on the exact same 80% split and predict on the 20% split.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training gene expression features (80%).
    y_train : pd.Series
        Training target diagnosis labels (80%).
    X_test : pd.DataFrame
        Testing gene expression features (20%, target hidden).
    classes : List[str]
        Sorted list of unique cancer diagnosis classes.
    mlp_hidden_layers : Tuple[int, ...], default=(128, 64)
        Hidden layer architecture for the Neural Network.
    mlp_max_iter : int, default=300
        Maximum epochs for the Neural Network.
    rf_n_estimators : int, default=200
        Number of trees for Random Forest.
    lr_c : float, default=1.0
        Inverse regularization strength for Logistic Regression.
    random_state : int, default=42
        Random seed for all three models.

    Returns
    -------
    Dict[str, ModelResult]
        Dictionary keyed by 'mlp', 'rf', and 'lr' mapping to their ModelResult objects.
    """
    # Fallback for tiny datasets (like the 30-row template) where early_stopping
    # validation split (10%) would have fewer samples than classes
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
