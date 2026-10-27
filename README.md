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

