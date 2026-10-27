"""Data loading, sample generation, and validation for RNA-seq expression data.
"""
"""Data loading, sample generation, and validation for RNA-seq expression data.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


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
    numeric_df = pd.DataFrame(index=df.index)

    for gene in gene_cols:
        converted = pd.to_numeric(df[gene], errors="coerce")
        if converted.isnull().any():
            non_numeric_cols.append(gene)
        else:
            numeric_df[gene] = converted

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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


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
    numeric_df = pd.DataFrame(index=df.index)

    for gene in gene_cols:
        converted = pd.to_numeric(df[gene], errors="coerce")
        if converted.isnull().any():
            non_numeric_cols.append(gene)
        else:
            numeric_df[gene] = converted

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

