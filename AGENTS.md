# AGENTS.md

## Project Overview
This project is a web-based bioinformatics application for classifying cancer types from RNA-seq gene expression data (such as TCGA). It compares a Deep Learning model (Multi-Layer Perceptron Neural Network) against classical machine learning baselines (Random Forest and Logistic Regression) and provides visual and biological interpretability.

## Technology Stack
- Language: Python 3.10+
- Frontend / UI: Streamlit
- Data Manipulation: Pandas, NumPy
- Machine Learning & Dimensionality Reduction: Scikit-learn, umap-learn
- Deep Learning: PyTorch (or Scikit-learn MLPClassifier with configurable hidden layers)
- Visualization: Plotly or Matplotlib / Seaborn

## Coding & Style Rules
- Do not use emojis anywhere in the codebase.
- Keep the code modular, clean, and easy for university students to read and maintain.
- Use type hints and short, clear docstrings for all functions.
- Handle errors gracefully in the Streamlit interface using `st.error`, `st.warning`, and `st.info`.

## Color Scheme & UI Theme
The application uses a clean clinical and genomics-inspired palette.

### Streamlit Theme (`.streamlit/config.toml`)
- Primary Accent Color: `#0284C7` (Clinical Sky Blue)
- Background Color: `#F8FAFC` (Light Alabaster Slate)
- Secondary Background Color: `#EFF6FF` (Soft Ice Blue)
- Text Color: `#0F172A` (Deep Slate)
- Font: `sans serif`

### Visualization Palette
- Categorical Palette (for PCA, UMAP, and ROC curves):
  `["#0284C7", "#0D9488", "#D97706", "#E11D48", "#7C3AED", "#2563EB"]`
- Heatmap Colormap (for Confusion Matrices): `"Blues"`
- Feature Importance Bar Color: `#0D9488` (Genomic Teal)

## Data Format Specification
The input dataset must be a CSV file structured as follows:
1. First column: Sample or Patient identifier (e.g., `Patient`).
2. Middle columns: Numeric gene expression values where each column header is a gene name (e.g., `TP53`, `BRCA1`, `EGFR`). All values must be numeric (`float` or `int`).
3. Last column: Target label representing the cancer diagnosis (e.g., `Diagnosis` or `Cancer_Type`).

Example table structure:
| Patient | TP53 | BRCA1 | EGFR | Diagnosis |
| :--- | :--- | :--- | :--- | :--- |
| Patient 1 | 12.4 | 1.1 | 0.4 | Breast Cancer |
| Patient 2 | 0.8 | 15.2 | 8.3 | Lung Cancer |
| Patient 3 | 11.9 | 0.9 | 1.2 | Breast Cancer |
| Patient 4 | 0.5 | 14.8 | 9.1 | Lung Cancer |
| Patient 5 | 13.1 | 1.0 | 0.2 | Breast Cancer |

## Application Workflow
The Streamlit application follows a strict step-by-step pipeline:

### Step 1: Data Upload, Template, and Exploratory Visualization
- Display clear instructions explaining the required CSV structure, including a visual sample table and a button to download a sample CSV template.
- Validate the uploaded CSV (check for missing values, verify that gene columns are numeric, and confirm at least two distinct diagnosis classes exist).
- Compute and display 2D dimensionality reduction scatter plots (PCA and UMAP) where each point is a patient colored by their cancer type.
- Provide a confirmation control allowing the user to inspect cluster separation and decide whether to proceed to model training.

### Step 2: Train / Test Split
- Split the dataset into 80% training data and 20% testing data using stratified sampling on the target column.
- Hide the target diagnosis column in the 20% test set during prediction and use it strictly for final evaluation.

### Step 3: Model Training
Train three distinct models on the exact same 80% training split:
1. Neural Network (Multi-Layer Perceptron).
2. Random Forest Classifier (`RandomForestClassifier` from Scikit-learn).
3. Logistic Regression (`LogisticRegression` from Scikit-learn).

### Step 4: Evaluation and Comparison
Evaluate all three models on the 20% test split and display for each model:
1. Accuracy: Displayed both as a percentage and as a ratio (e.g., `93 / 100 patients (93.0% accuracy)`).
2. Confusion Matrix: 2D heatmap with true diagnoses on the rows and predicted diagnoses on the columns.
3. ROC Curves: One-vs-Rest ROC curves from `(0, 0)` to `(1, 1)` showing the trade-off between true positive rate and false positive rate for each cancer type, including AUC scores.
4. Feature Importance (Top Genes): Horizontal bar chart showing the top contributing genes driving the classification decisions.

## Repository Directory Structure
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