# GenoScope: RNA-seq Cancer Classification

Web-based bioinformatics application for classifying cancer types from RNA-seq gene expression data (e.g., TCGA transcriptomics). The application benchmarks a Deep Learning Multi-Layer Perceptron (MLP) against classical machine learning baselines (Random Forest and Logistic Regression) and provides biological interpretability.

## Technology Stack
- Language: Python 3.10+
- Web Framework: Streamlit
- Data Manipulation: Pandas, NumPy
- Machine Learning & Dimensionality Reduction: Scikit-learn, umap-learn
- Interactive Visualization: Plotly

## Project Structure
```text
.
├── AGENTS.md
├── README.md
├── requirements.txt
├── app.py
├── .streamlit/
│   └── config.toml
├── data/
│   └── sample_template.csv
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── visualization.py
    ├── models.py
    └── evaluation.py
```

## Setup Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Launch the Streamlit application:
```bash
streamlit run app.py
```

## Dataset Specification
Input data must be supplied as a CSV file following this structure:
1. First column: Unique Patient or sample identifier (e.g., `Patient`).
2. Middle columns: Numeric gene expression values (e.g., `TP53`, `BRCA1`, `EGFR`, `MYC`).
3. Last column: Ground-truth cancer diagnosis label (e.g., `Diagnosis`).

A ready-to-use template with 30 synthetic patient profiles across 3 cancer classes is available at `data/sample_template.csv` and can also be downloaded directly within the application.

## Workflow Overview
- Step 1: Data Upload, Template, and Exploratory Visualization (PCA and UMAP)
- Step 2: Stratified 80/20 Train / Test Split (Held-out evaluation set)
- Step 3: Model Training (MLP Neural Network, Random Forest, Logistic Regression)
- Step 4: Model Evaluation and Biological Interpretability (Accuracy, Confusion Matrix, ROC curves, Top Gene Features)

## Developer Guide: Accessing Split Tables (Steps 3 & 4)

### 1. In Streamlit (`app.py`)
Teammates implementing subsequent steps in Streamlit can access the partitioned tables directly from session state:

```python
# Training inputs for Step 3:
X_train = st.session_state["X_train"]  # pd.DataFrame: shape (N_train, N_genes)
y_train = st.session_state["y_train"]  # pd.Series: shape (N_train,)
train_df = st.session_state["train_df"]  # pd.DataFrame: complete 80% table

# Testing inputs for Step 4 (Target Diagnosis strictly hidden):
X_test = st.session_state["X_test"]    # pd.DataFrame: shape (N_test, N_genes)
y_test = st.session_state["y_test"]    # pd.Series: shape (N_test,), ground truth labels
test_df = st.session_state["test_df"]  # pd.DataFrame: complete 20% table with labels

# Split metadata:
split_data = st.session_state["split_data"]
classes = split_data["classes"]        # List of cancer diagnoses
gene_cols = split_data["gene_cols"]    # List of gene names
```

### 2. Programmatically in Python (`src/models.py` or `src/evaluation.py`)
```python
from src.data_loader import load_dataset_from_csv, validate_dataset, split_dataset_from_metadata

# 1. Load and validate dataset
df = load_dataset_from_csv("data/sample_template.csv")
is_valid, _, metadata = validate_dataset(df)

# 2. Perform 80/20 stratified split
split_data = split_dataset_from_metadata(metadata, test_size=0.20, random_state=42)

X_train, y_train = split_data["X_train"], split_data["y_train"]
X_test, y_test = split_data["X_test"], split_data["y_test"]
```
