"""Data loading, sample generation, validation, and train/test splitting for RNA-seq expression data.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_sample_dataset(
    n_samples_per_class: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """Generate a realistic synthetic RNA-seq gene expression dataset.

    Parameters
    ----------
    n_samples_per_class : int, default=10
        Number of patient samples to generate for each cancer diagnosis.
    random_state : int, default=42
        Seed for reproducible random noise generation.

    Returns
    -------
    pd.DataFrame
        DataFrame structured as:
        - Patient: string patient identifier
        - Gene expression columns: TP53, BRCA1, EGFR, MYC, KRAS, PTEN, ERBB2, PIK3CA
        - Diagnosis: target cancer diagnosis label
    """
    rng = np.random.default_rng(random_state)

    genes = ["TP53", "BRCA1", "EGFR", "MYC", "KRAS", "PTEN", "ERBB2", "PIK3CA"]

    classes = [
        "Breast Cancer",
        "Lung Adenocarcinoma",
        "Colorectal Cancer",
    ]

    # Baseline log2 expression signatures per cancer type
    profiles = {
        "Breast Cancer": {
            "TP53": 5.2, "BRCA1": 15.4, "EGFR": 2.1, "MYC": 6.0,
            "KRAS": 3.8, "PTEN": 4.1, "ERBB2": 14.8, "PIK3CA": 11.7,
        },
        "Lung Adenocarcinoma": {
            "TP53": 11.8, "BRCA1": 2.4, "EGFR": 15.1, "MYC": 5.4,
            "KRAS": 13.9, "PTEN": 3.5, "ERBB2": 2.6, "PIK3CA": 4.3,
        },
        "Colorectal Cancer": {
            "TP53": 12.5, "BRCA1": 2.1, "EGFR": 4.8, "MYC": 14.2,
            "KRAS": 12.1, "PTEN": 2.9, "ERBB2": 1.9, "PIK3CA": 5.0,
        },
    }

    records = []
    patient_counter = 1

    for cancer_type in classes:
        profile = profiles[cancer_type]
        for _ in range(n_samples_per_class):
            row_dict: Dict[str, Any] = {
                "Patient": f"Patient_{patient_counter:03d}"
            }
            for gene in genes:
                base_expr = profile[gene]
                noise = rng.normal(loc=0.0, scale=0.65)
                # Keep expression non-negative
                expression = round(max(0.1, base_expr + noise), 2)
                row_dict[gene] = expression
            row_dict["Diagnosis"] = cancer_type
            records.append(row_dict)
            patient_counter += 1

    return pd.DataFrame(records)


def save_sample_template(
    output_path: Union[str, Path] = "data/sample_template.csv"
) -> Path:
    """Save the synthetic RNA-seq expression dataset as a CSV template.

    Parameters
    ----------
    output_path : Union[str, Path], default='data/sample_template.csv'
        Destination file path.

    Returns
    -------
    Path
        Absolute path to the created CSV template.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_sample_dataset(n_samples_per_class=10, random_state=42)
    df.to_csv(path, index=False)
    return path.resolve()


def load_dataset_from_csv(file_or_path: Any) -> pd.DataFrame:
    """Read a CSV file into a pandas DataFrame.

    Parameters
    ----------
    file_or_path : Any
        File-like object (e.g., Streamlit UploadedFile) or file path string/Path.

    Returns
    -------
    pd.DataFrame
        Loaded raw DataFrame.
    """
    return pd.read_csv(file_or_path)


def validate_dataset(
    df: pd.DataFrame
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Validate that the input DataFrame complies with the required RNA-seq structure.

    The expected structure is:
    1. First column: Patient or sample identifier.
    2. Middle columns: Numeric gene expression values.
    3. Last column: Cancer diagnosis class label.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    Returns
    -------
    Tuple[bool, str, Optional[Dict[str, Any]]]
        - is_valid (bool): True if dataset passes all validation checks.
        - message (str): Explanation of errors or success confirmation.
        - data_info (dict or None): Extracted dataset metadata if valid.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return False, "Input data is not a valid pandas DataFrame.", None

    if df.empty:
        return False, "The uploaded CSV file is empty.", None

    # Check minimum columns: 1 Patient ID + at least 1 gene feature + 1 target
    if df.shape[1] < 3:
        return (
            False,
            f"Dataset requires at least 3 columns (Patient ID, at least one Gene, and Diagnosis). "
            f"Found {df.shape[1]} columns.",
            None,
        )

    # Check minimum rows for downstream processing
    if df.shape[0] < 4:
        return (
            False,
            f"Dataset has only {df.shape[0]} rows. At least 4 rows are required for analysis.",
            None,
        )

    col_names = list(df.columns)
    patient_col = col_names[0]
    target_col = col_names[-1]
    gene_cols = col_names[1:-1]

    # Check for missing/null values across the dataset
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        null_details = ", ".join(
            f"{col} ({count} missing)" for col, count in cols_with_nulls.items()
        )
        return (
            False,
            f"Dataset contains missing values in columns: {null_details}. "
            f"Please clean or impute missing values before uploading.",
            None,
        )

    # Validate that middle columns are purely numeric
    non_numeric_cols: List[str] = []
    non_numeric_candidates = [
        col for col in gene_cols if not pd.api.types.is_numeric_dtype(df[col])
    ]

    if non_numeric_candidates:
        numeric_df = df[gene_cols].copy()
        for gene in non_numeric_candidates:
            converted = pd.to_numeric(numeric_df[gene], errors="coerce")
            if converted.isnull().any():
                non_numeric_cols.append(gene)
            else:
                numeric_df[gene] = converted
    else:
        numeric_df = df[gene_cols]

    if non_numeric_cols:
        cols_formatted = ", ".join(non_numeric_cols[:5])
        suffix = "..." if len(non_numeric_cols) > 5 else ""
        return (
            False,
            f"All middle gene columns must contain numeric values. "
            f"Found non-numeric values in: {cols_formatted}{suffix}.",
            None,
        )

    # Validate target diagnosis column
    target_series = df[target_col].astype(str).str.strip()
    unique_classes = target_series.unique().tolist()

    if len(unique_classes) < 2:
        return (
            False,
            f"Target column '{target_col}' has only {len(unique_classes)} class "
            f"({unique_classes}). Classification requires at least 2 distinct cancer diagnoses.",
            None,
        )

    # Verify each class has at least 2 samples to allow stratified splitting
    class_distribution = target_series.value_counts().to_dict()
    scarce_classes = [cls for cls, count in class_distribution.items() if count < 2]
    if scarce_classes:
        return (
            False,
            f"The following classes have fewer than 2 samples: {', '.join(scarce_classes)}. "
            f"Each cancer diagnosis must have at least 2 samples for stratified train/test splitting.",
            None,
        )

    # Standardized clean representation
    clean_df = pd.concat([
        df[[patient_col]].astype(str),
        numeric_df,
        target_series.to_frame(name=target_col)
    ], axis=1)

    metadata: Dict[str, Any] = {
        "patient_col": patient_col,
        "gene_cols": gene_cols,
        "target_col": target_col,
        "n_patients": clean_df.shape[0],
        "n_genes": len(gene_cols),
        "classes": sorted(unique_classes),
        "class_distribution": class_distribution,
        "features": numeric_df,
        "labels": target_series,
        "patient_ids": clean_df[patient_col],
        "clean_df": clean_df,
    }

    return True, "Dataset validated successfully.", metadata


def split_train_test(
    clean_df: pd.DataFrame,
    patient_col: str,
    gene_cols: List[str],
    target_col: str,
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True,
) -> Dict[str, Any]:
    """Split the cleaned RNA-seq dataset into stratified train and test partitions.

    In accordance with bioinformatics best practices and AGENTS.md:
    1. 80% training data and 20% testing data (default test_size=0.20).
    2. Stratified sampling on the target cancer diagnosis column ensures identical
       class proportions across both splits.
    3. The target diagnosis column is hidden in the test features table for model prediction,
       and preserved in a separate ground truth structure strictly for final evaluation.

    Parameters
    ----------
    clean_df : pd.DataFrame
        Cleaned DataFrame containing patient IDs, numeric gene expressions, and diagnosis labels.
    patient_col : str
        Column name representing the patient/sample identifier.
    gene_cols : List[str]
        List of numeric gene expression feature column names.
    target_col : str
        Column name representing the target cancer diagnosis.
    test_size : float, default=0.20
        Fraction of the dataset to allocate to the held-out test split (0.20 = 20%).
    random_state : int, default=42
        Seed for reproducible shuffling and stratified partitioning.
    stratify : bool, default=True
        Whether to perform stratified sampling based on the target column.

    Returns
    -------
    Dict[str, Any]
        Dictionary with standard machine learning and bioinformatics tables:
        - 'X_train': pd.DataFrame with training gene expression features (80%).
        - 'y_train': pd.Series with training diagnosis labels (80%).
        - 'X_test': pd.DataFrame with testing gene expression features (20%, target excluded).
        - 'y_test': pd.Series with true test diagnosis labels (20%, strictly for evaluation).
        - 'train_patient_ids': pd.Series with patient IDs in training set.
        - 'test_patient_ids': pd.Series with patient IDs in test set.
        - 'train_df': pd.DataFrame with full training records (Patient, Genes, Diagnosis).
        - 'test_features_df': pd.DataFrame with test records where target is hidden (Patient, Genes).
        - 'test_labels_df': pd.DataFrame with ground truth test labels (Patient, Diagnosis).
        - 'test_df': pd.DataFrame with full test records (Patient, Genes, Diagnosis).
        - 'classes': List[str] of sorted distinct diagnosis categories.
        - 'train_size': int count of training samples.
        - 'test_size': int count of testing samples.
        - 'test_ratio': float test split fraction.
        - 'random_state': int random seed used.
        - 'class_distribution_train': Dict[str, int] class counts in training split.
        - 'class_distribution_test': Dict[str, int] class counts in test split.
        - 'gene_cols': List[str] of gene column names.
        - 'patient_col': str patient column name.
        - 'target_col': str target column name.
    """
    if clean_df is None or clean_df.empty:
        raise ValueError("Cannot split an empty or None DataFrame.")

    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be strictly between 0 and 1, got {test_size}.")

    stratify_target = clean_df[target_col] if stratify else None

    try:
        train_df, test_df = train_test_split(
            clean_df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_target,
        )
    except ValueError as exc:
        # Fall back to unstratified split if class counts cannot support stratification
        if stratify:
            train_df, test_df = train_test_split(
                clean_df,
                test_size=test_size,
                random_state=random_state,
                stratify=None,
            )
        else:
            raise exc

    # Reset index for clean contiguous indexing
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Separate feature matrices and target vectors
    X_train = train_df[gene_cols].copy()
    y_train = train_df[target_col].copy()
    train_patient_ids = train_df[patient_col].copy()

    X_test = test_df[gene_cols].copy()
    y_test = test_df[target_col].copy()
    test_patient_ids = test_df[patient_col].copy()

    # Create unlabelled test features table (target hidden for blind prediction)
    test_features_df = test_df[[patient_col] + gene_cols].copy()

    # Ground truth test labels table for evaluation
    test_labels_df = test_df[[patient_col, target_col]].copy()

    unique_classes = sorted(clean_df[target_col].unique().tolist())
    class_dist_train = y_train.value_counts().to_dict()
    class_dist_test = y_test.value_counts().to_dict()

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "train_patient_ids": train_patient_ids,
        "test_patient_ids": test_patient_ids,
        "train_df": train_df,
        "test_features_df": test_features_df,
        "test_labels_df": test_labels_df,
        "test_df": test_df,
        "classes": unique_classes,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "test_ratio": test_size,
        "random_state": random_state,
        "class_distribution_train": class_dist_train,
        "class_distribution_test": class_dist_test,
        "gene_cols": gene_cols,
        "patient_col": patient_col,
        "target_col": target_col,
    }


def split_dataset_from_metadata(
    metadata: Dict[str, Any],
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True,
) -> Dict[str, Any]:
    """Convenience helper to split dataset directly from validate_dataset metadata.

    Parameters
    ----------
    metadata : Dict[str, Any]
        Metadata dictionary returned by validate_dataset().
    test_size : float, default=0.20
        Fraction of dataset for testing split.
    random_state : int, default=42
        Random seed for reproducibility.
    stratify : bool, default=True
        Whether to perform stratified sampling.

    Returns
    -------
    Dict[str, Any]
        The complete split data dictionary produced by split_train_test.
    """
    return split_train_test(
        clean_df=metadata["clean_df"],
        patient_col=metadata["patient_col"],
        gene_cols=metadata["gene_cols"],
        target_col=metadata["target_col"],
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
