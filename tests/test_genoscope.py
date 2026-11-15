"""Unit test suite for GenoScope bioinformatics pipeline.
Run with: pytest
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    generate_sample_dataset,
    split_dataset_from_metadata,
    validate_dataset,
)
from src.evaluation import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curves,
)
from src.models import (
    extract_feature_importances,
    predict_single_patient_df,
    train_all_models,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Fixture providing a 30-patient, 8-gene synthetic RNA-seq dataset."""
    return generate_sample_dataset(n_samples_per_class=10, random_state=42)


@pytest.fixture
def split_bundle(sample_df: pd.DataFrame):
    """Fixture providing validated metadata and 80/20 stratified split."""
    is_valid, msg, metadata = validate_dataset(sample_df)
    assert is_valid, f"Sample dataset failed validation: {msg}"
    assert metadata is not None
    return split_dataset_from_metadata(metadata, test_size=0.20, random_state=42)


def test_validate_dataset_success(sample_df: pd.DataFrame) -> None:
    """Verify valid dataset passes all structural and numeric checks."""
    is_valid, msg, metadata = validate_dataset(sample_df)
    assert is_valid is True
    assert metadata is not None
    assert metadata["n_patients"] == 30
    assert metadata["n_genes"] == 8
    assert len(metadata["classes"]) == 3


def test_validate_dataset_rejects_missing_values(sample_df: pd.DataFrame) -> None:
    """Verify dataset with NaN gene expression values is rejected."""
    corrupted_df = sample_df.copy()
    corrupted_df.loc[0, "TP53"] = np.nan
    is_valid, msg, metadata = validate_dataset(corrupted_df)
    assert is_valid is False
    assert "missing values" in msg.lower()
    assert metadata is None


def test_stratified_split_and_blind_test_protection(split_bundle) -> None:
    """Verify 80/20 split preserves class proportions and hides Diagnosis from X_test."""
    assert split_bundle["train_size"] == 24
    assert split_bundle["test_size"] == 6
    assert "Diagnosis" not in split_bundle["X_test"].columns
    assert "Diagnosis" not in split_bundle["test_features_df"].columns
    assert list(split_bundle["X_train"].columns) == split_bundle["gene_cols"]


def test_train_all_models_and_shapes(split_bundle) -> None:
    """Verify MLP, Random Forest, and Logistic Regression train and output valid shapes."""
    results = train_all_models(
        X_train=split_bundle["X_train"],
        y_train=split_bundle["y_train"],
        X_test=split_bundle["X_test"],
        classes=split_bundle["classes"],
        mlp_hidden_layers=(16, 8),
        mlp_max_iter=100,
        rf_n_estimators=50,
        random_state=42,
    )

    assert set(results.keys()) == {"mlp", "rf", "lr"}
    n_test = split_bundle["test_size"]
    n_classes = len(split_bundle["classes"])
    n_genes = len(split_bundle["gene_cols"])

    for key, res in results.items():
        assert res.y_pred.shape == (n_test,)
        assert res.y_prob.shape == (n_test, n_classes)
        assert np.allclose(np.sum(res.y_prob, axis=1), 1.0, atol=1e-5)
        assert res.feature_importances.shape == (n_genes,)
        assert np.all(res.feature_importances >= 0.0)


def test_evaluation_metrics_and_charts(split_bundle) -> None:
    """Verify accuracy metrics and Plotly figures generate without errors."""
    results = train_all_models(
        X_train=split_bundle["X_train"],
        y_train=split_bundle["y_train"],
        X_test=split_bundle["X_test"],
        classes=split_bundle["classes"],
        mlp_hidden_layers=(16, 8),
        rf_n_estimators=20,
        random_state=42,
    )
    y_test = split_bundle["y_test"]
    classes = split_bundle["classes"]
    gene_cols = split_bundle["gene_cols"]

    for key, res in results.items():
        metrics = calculate_metrics(y_test, res.y_pred)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["total_count"] == len(y_test)
        assert "patients" in metrics["formatted_accuracy"]

        fig_cm = plot_confusion_matrix(y_test, res.y_pred, classes, model_name=res.model_name)
        assert fig_cm is not None

        fig_roc = plot_roc_curves(y_test, res.y_prob, classes, model_name=res.model_name)
        assert fig_roc is not None

        fig_imp = plot_feature_importance(res.estimator, gene_cols, top_n=5, model_name=res.model_name)
        assert fig_imp is not None


def test_single_patient_inference_with_and_without_diagnosis(split_bundle) -> None:
    """Verify single-patient CSV inference works both with ground-truth label and blind."""
    results = train_all_models(
        X_train=split_bundle["X_train"],
        y_train=split_bundle["y_train"],
        X_test=split_bundle["X_test"],
        classes=split_bundle["classes"],
        mlp_hidden_layers=(16, 8),
        rf_n_estimators=20,
        random_state=42,
    )

    # 1. With ground-truth label (last row of 20% test_df)
    single_labelled_df = split_bundle["test_df"].tail(1)
    report_labelled = predict_single_patient_df(
        patient_df=single_labelled_df,
        model_results=results,
        gene_cols=split_bundle["gene_cols"],
        patient_col=split_bundle["patient_col"],
        target_col=split_bundle["target_col"],
    )
    assert report_labelled["true_diagnosis"] is not None
    for key in ["mlp", "rf", "lr"]:
        assert report_labelled["predictions"][key]["is_correct"] in (True, False)

    # 2. Blind (without Diagnosis column)
    single_blind_df = split_bundle["test_features_df"].tail(1)
    report_blind = predict_single_patient_df(
        patient_df=single_blind_df,
        model_results=results,
        gene_cols=split_bundle["gene_cols"],
        patient_col=split_bundle["patient_col"],
        target_col=split_bundle["target_col"],
    )
    assert report_blind["true_diagnosis"] is None
    for key in ["mlp", "rf", "lr"]:
        assert report_blind["predictions"][key]["is_correct"] is None