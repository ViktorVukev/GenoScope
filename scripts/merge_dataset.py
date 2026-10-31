"""Merge RNA-seq expression data and labels into a single unified CSV file.

Merges data.csv (expression matrix) and labels.csv (cancer diagnoses) into:
data/TCGA-PANCAN-HiSeq-801x20531.csv
where each row contains the sample identifier, 20531 gene expression values,
and the cancer diagnosis type appended as the final column.
"""

from pathlib import Path
from typing import Dict, Optional
import time


def merge_data_and_labels(
    data_path: str = "data/data.csv",
    labels_path: str = "data/labels.csv",
    output_path: str = "data/TCGA-PANCAN-HiSeq-801x20531.csv",
) -> Path:
    """Stream-merge gene expression matrix with diagnosis labels.

    Parameters
    ----------
    data_path : str, default='data/data.csv'
        Path to expression matrix CSV file.
    labels_path : str, default='data/labels.csv'
        Path to cancer diagnosis labels CSV file.
    output_path : str, default='data/TCGA-PANCAN-HiSeq-801x20531.csv'
        Destination path for merged CSV file.

    Returns
    -------
    Path
        Absolute path to the created unified CSV file.
    """
    data_file = Path(data_path)
    labels_file = Path(labels_path)
    out_file = Path(output_path)

    if not data_file.exists():
        raise FileNotFoundError(f"Expression matrix file not found: {data_file}")
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")

    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Parse labels mapping: sample_id -> cancer_type
    labels: Dict[str, str] = {}
    label_col_name = "Class"
    with open(labels_file, "r", encoding="utf-8") as f_labels:
        header = f_labels.readline().strip().split(",")
        if len(header) > 1 and header[1]:
            label_col_name = header[1]
        for line in f_labels:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                labels[parts[0]] = parts[1]

    # 2. Stream merge data with labels row by row
    matched_count = 0
    with open(data_file, "r", encoding="utf-8") as f_in, open(out_file, "w", encoding="utf-8") as f_out:
        # Header
        raw_header = f_in.readline().rstrip("\r\n")
        header_cols = raw_header.split(",")
        if header_cols[0] == "":
            header_cols[0] = "Sample"
        header_cols.append(label_col_name)
        f_out.write(",".join(header_cols) + "\n")

        # Data rows
        for line in f_in:
            line_clean = line.rstrip("\r\n")
            sample_id = line_clean.split(",", 1)[0]
            if sample_id in labels:
                cancer_type = labels[sample_id]
                f_out.write(f"{line_clean},{cancer_type}\n")
                matched_count += 1

    return out_file.resolve()


if __name__ == "__main__":
    start_time = time.time()
    result_path = merge_data_and_labels()
    elapsed = time.time() - start_time
    print(f"Successfully generated merged dataset: {result_path} in {elapsed:.2f} seconds.")
