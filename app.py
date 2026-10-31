"""GenoScope: Web-based bioinformatics application for classifying cancer types
from RNA-seq gene expression data.
Step 1: Data Upload, Template, and Exploratory Visualization.
Step 2: Stratified Train / Test Split (80% Training / 20% Testing).
"""

from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import streamlit as st

from src.data_loader import (
    generate_sample_dataset,
    load_dataset_from_csv,
    save_sample_template,
    split_dataset_from_metadata,
    validate_dataset,
)
from src.visualization import plot_pca, plot_umap

# Configure page layout and metadata
st.set_page_config(
    page_title="GenoScope - RNA-seq Cancer Classification",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Clinical custom styling adhering to AGENTS.md theme specifications
st.markdown(
    """
    <style>
    .main {
        background-color: #F8FAFC;
    }
    .header-box {
        background-color: #FFFFFF;
        border-left: 5px solid #0284C7;
        padding: 20px 24px;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
        margin-bottom: 24px;
    }
    .header-title {
        color: #0F172A;
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .header-subtitle {
        color: #475569;
        font-size: 15px;
        line-height: 1.5;
    }
    .step-badge {
        display: inline-block;
        padding: 4px 10px;
        background-color: #EFF6FF;
        color: #0284C7;
        font-weight: 600;
        font-size: 13px;
        border-radius: 4px;
        border: 1px solid #BFDBFE;
        margin-bottom: 8px;
    }
    .section-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 24px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .info-callout {
        background-color: #EFF6FF;
        border-left: 4px solid #0284C7;
        padding: 12px 16px;
        border-radius: 4px;
        color: #0F172A;
        font-size: 14px;
        margin: 14px 0;
    }
    .success-callout {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
        padding: 12px 16px;
        border-radius: 4px;
        color: #14532D;
        font-size: 14px;
        margin: 14px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def ensure_template_exists() -> Path:
    """Ensure the template CSV file exists on disk and return its path.

    Returns
    -------
    Path
        Path to the template CSV file.
    """
    template_path = Path("data/sample_template.csv")
    if not template_path.exists():
        save_sample_template(template_path)
    return template_path


def render_header(current_step: int = 1) -> None:
    """Render top application header and workflow pipeline summary.

    Parameters
    ----------
    current_step : int, default=1
        The current active stage in the 4-step workflow.
    """
    stage_titles = {
        1: "Stage 1 of 4: Exploratory Analysis",
        2: "Stage 2 of 4: Stratified Train / Test Split",
        3: "Stage 3 of 4: Model Training",
        4: "Stage 4 of 4: Evaluation and Interpretability",
    }
    badge_text = stage_titles.get(current_step, "Stage 1 of 4: Exploratory Analysis")

    st.markdown(
        f"""
        <div class="header-box">
            <span class="step-badge">{badge_text}</span>
            <div class="header-title">GenoScope: RNA-seq Cancer Classification and Biological Interpretability</div>
            <div class="header-subtitle">
                Benchmark Deep Learning (Multi-Layer Perceptron) against classical machine learning
                baselines (Random Forest and Logistic Regression) on transcriptomic profiles.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_format_instructions(template_path: Path) -> None:
    """Render data specification instructions, visual table preview, and download controls.

    Parameters
    ----------
    template_path : Path
        Path to the template CSV file.
    """
    st.markdown("### Step 1: Data Upload, Template, and Exploratory Visualization")
    st.write(
        "To begin analysis, upload a normalized RNA-seq gene expression dataset in CSV format. "
        "The dataset must strictly adhere to the following column hierarchy:"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            "**1. First Column: Patient ID**\n\n"
            "Unique identifier for each clinical sample (e.g., `Patient_001`, `TCGA-BRCA-01`)."
        )
    with col2:
        st.markdown(
            "**2. Middle Columns: Gene Expression**\n\n"
            "Normalized numeric values (e.g., log2 RPKM/TPM) for each gene (e.g., `TP53`, `BRCA1`)."
        )
    with col3:
        st.markdown(
            "**3. Last Column: Cancer Diagnosis**\n\n"
            "Categorical ground-truth cancer diagnosis or subtype (e.g., `Breast Cancer`)."
        )

    st.markdown("**Required CSV Layout Preview:**")
    sample_preview_df = pd.DataFrame(
        [
            ["Patient_001", 5.40, 14.72, 2.59, 6.61, "Breast Cancer"],
            ["Patient_002", 12.10, 1.97, 14.86, 5.15, "Lung Adenocarcinoma"],
            ["Patient_003", 11.87, 1.63, 6.18, 13.67, "Colorectal Cancer"],
        ],
        columns=["Patient", "TP53", "BRCA1", "EGFR", "MYC", "Diagnosis"],
    )
    st.dataframe(sample_preview_df, use_container_width=True, hide_index=True)

    btn_col1, btn_col2, _ = st.columns([1, 1, 2])

    with btn_col1:
        if template_path.exists():
            with open(template_path, "rb") as f:
                template_bytes = f.read()
        else:
            template_bytes = generate_sample_dataset().to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download CSV Template",
            data=template_bytes,
            file_name="sample_template.csv",
            mime="text/csv",
            key="download_csv_template_btn",
            help="Download a pre-formatted template with 30 sample patients and 8 genes.",
        )

    with btn_col2:
        if st.button(
            "Load Built-in Sample Data",
            key="load_sample_data_btn",
            help="Quickly test the app using the template dataset.",
        ):
            st.session_state["raw_df"] = load_dataset_from_csv(template_path)
            st.session_state["source_label"] = "Built-in Sample Template (30 patients, 8 genes, 3 classes)"
            st.session_state["current_step"] = 1
            st.session_state["split_data"] = None


def render_step2_split(metadata: Dict[str, Any]) -> None:
    """Render Step 2: Stratified Train / Test Split and developer integration tables.

    Parameters
    ----------
    metadata : Dict[str, Any]
        Metadata dictionary extracted from validated dataset.
    """
    st.markdown("### Step 2: Stratified Train / Test Split (80% / 20%)")
    st.markdown(
        """
        <div class="info-callout">
            The dataset is partitioned into an <b>80% training split</b> for model optimization
            and a <b>20% held-out test split</b> for unbiased performance evaluation.
            Stratified sampling ensures the exact proportions of each cancer diagnosis are preserved across both subsets.
        </div>
        """,
        unsafe_allow_html=True,
    )

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 1])

    with ctrl_col1:
        st.markdown("**Split Ratio:** 80% Train / 20% Test")
        test_ratio = 0.20

    with ctrl_col2:
        seed_val = st.number_input(
            "Random Seed (Reproducibility)",
            min_value=0,
            max_value=99999,
            value=42,
            step=1,
            key="split_random_seed",
            help="Seed ensuring deterministic, reproducible sample partitioning across team members.",
        )

    with ctrl_col3:
        st.write("")
        st.write("")
        recompute_btn = st.button("Recompute 80/20 Split", key="recompute_split_btn")

    # Compute or retrieve split
    need_computation = (
        "split_data" not in st.session_state
        or st.session_state["split_data"] is None
        or recompute_btn
    )

    if need_computation:
        try:
            split_data = split_dataset_from_metadata(
                metadata=metadata,
                test_size=test_ratio,
                random_state=int(seed_val),
                stratify=True,
            )
            st.session_state["split_data"] = split_data

            # Store direct references in session state for teammates working on models and evaluation
            st.session_state["X_train"] = split_data["X_train"]
            st.session_state["y_train"] = split_data["y_train"]
            st.session_state["X_test"] = split_data["X_test"]
            st.session_state["y_test"] = split_data["y_test"]
            st.session_state["train_df"] = split_data["train_df"]
            st.session_state["test_df"] = split_data["test_df"]
            st.session_state["test_features_df"] = split_data["test_features_df"]
            st.session_state["test_labels_df"] = split_data["test_labels_df"]
        except Exception as exc:
            st.error(f"Error computing stratified split: {exc}")
            return
    else:
        split_data = st.session_state["split_data"]

    # Metrics overview
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    train_pct = (1.0 - split_data["test_ratio"]) * 100.0
    test_pct = split_data["test_ratio"] * 100.0
    m_col1.metric("Training Samples", f"{split_data['train_size']} patients", f"{train_pct:.0f}% of total")
    m_col2.metric("Testing Samples", f"{split_data['test_size']} patients", f"{test_pct:.0f}% of total")
    m_col3.metric("Gene Features", f"{len(split_data['gene_cols'])} genes")
    m_col4.metric("Diagnosis Classes", f"{len(split_data['classes'])} classes")

    # Stratification Verification Table
    st.markdown("#### Stratification Verification")
    st.caption("Verification that relative class frequencies are identically maintained across both partitions:")

    balance_rows = []
    total_patients = split_data["train_size"] + split_data["test_size"]
    for cancer_class in split_data["classes"]:
        n_train = split_data["class_distribution_train"].get(cancer_class, 0)
        n_test = split_data["class_distribution_test"].get(cancer_class, 0)
        n_total_class = n_train + n_test
        class_pct_in_train = (n_train / split_data["train_size"] * 100.0) if split_data["train_size"] > 0 else 0.0
        class_pct_in_test = (n_test / split_data["test_size"] * 100.0) if split_data["test_size"] > 0 else 0.0

        balance_rows.append({
            "Diagnosis Class": cancer_class,
            "Total Patients": n_total_class,
            "Train Count (80%)": n_train,
            "Train Proportion": f"{class_pct_in_train:.1f}%",
            "Test Count (20%)": n_test,
            "Test Proportion": f"{class_pct_in_test:.1f}%",
        })

    balance_df = pd.DataFrame(balance_rows)
    st.dataframe(balance_df, use_container_width=True, hide_index=True)

    # Blind Prediction Callout
    st.markdown(
        """
        <div class="success-callout">
            <b>Blind Prediction Protection Enforced:</b><br>
            The target cancer diagnosis column has been hidden from the 20% test features table
            (<code>X_test</code> and <code>test_features_df</code>).
            Models receive only patient identifiers and gene expressions during inference.
            Ground-truth diagnosis labels are preserved separately in <code>y_test</code> strictly for Step 4 evaluation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Split Tables Inspection and Download Tabs
    st.markdown("#### Inspect and Download Partitioned Tables")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Training Set (80% with Labels)",
        "Test Features (20% Target Hidden)",
        "Test Ground Truth (20% for Evaluation)",
        "Full Test Set (20% Complete)",
    ])

    with tab1:
        st.markdown(
            f"**Training Set (`train_df` / `X_train`, `y_train`):** "
            f"{split_data['train_size']} patients, {len(split_data['gene_cols'])} gene features, plus Diagnosis label."
        )
        st.dataframe(split_data["train_df"], use_container_width=True, hide_index=True)
        st.download_button(
            label="Download Training Set CSV (80%)",
            data=split_data["train_df"].to_csv(index=False).encode("utf-8"),
            file_name="genoscope_train_80pct.csv",
            mime="text/csv",
            key="download_train_csv_btn",
        )

    with tab2:
        st.markdown(
            f"**Test Features (`test_features_df` / `X_test`):** "
            f"{split_data['test_size']} patients, {len(split_data['gene_cols'])} gene features. "
            f"Diagnosis column is strictly withheld."
        )
        st.dataframe(split_data["test_features_df"], use_container_width=True, hide_index=True)
        st.download_button(
            label="Download Test Features CSV (20% Target Hidden)",
            data=split_data["test_features_df"].to_csv(index=False).encode("utf-8"),
            file_name="genoscope_test_features_unlabelled_20pct.csv",
            mime="text/csv",
            key="download_test_features_csv_btn",
        )

    with tab3:
        st.markdown(
            f"**Test Ground Truth (`test_labels_df` / `y_test`):** "
            f"{split_data['test_size']} patients with true diagnoses for final benchmarking."
        )
        st.dataframe(split_data["test_labels_df"], use_container_width=True, hide_index=True)
        st.download_button(
            label="Download Test Labels CSV (20% Ground Truth)",
            data=split_data["test_labels_df"].to_csv(index=False).encode("utf-8"),
            file_name="genoscope_test_ground_truth_labels_20pct.csv",
            mime="text/csv",
            key="download_test_labels_csv_btn",
        )

    with tab4:
        st.markdown(
            f"**Complete Test Set (`test_df`):** "
            f"{split_data['test_size']} patients with all genes and diagnosis label."
        )
        st.dataframe(split_data["test_df"], use_container_width=True, hide_index=True)
        st.download_button(
            label="Download Full Test Set CSV (20%)",
            data=split_data["test_df"].to_csv(index=False).encode("utf-8"),
            file_name="genoscope_test_full_20pct.csv",
            mime="text/csv",
            key="download_test_full_csv_btn",
        )

    # Teammate Developer Reference Guide
    with st.expander("Developer Guide: How Teammates Can Access These Tables", expanded=True):
        st.markdown(
            """
            This section specifies how teammates developing **Step 3 (`src/models.py`)** and
            **Step 4 (`src/evaluation.py`)** can directly consume these partitions.

            ---

            ##### 1. Accessing via Streamlit Session State (Inside `app.py`)
            Teammates implementing subsequent steps in Streamlit can access the split tables directly:
            ```python
            # Training inputs for Step 3 (Neural Network, Random Forest, Logistic Regression):
            X_train = st.session_state["X_train"]  # pd.DataFrame: shape (N_train, N_genes)
            y_train = st.session_state["y_train"]  # pd.Series: shape (N_train,)
            train_df = st.session_state["train_df"]  # pd.DataFrame: complete 80% table

            # Testing inputs for Step 4 (Model Evaluation):
            X_test = st.session_state["X_test"]    # pd.DataFrame: shape (N_test, N_genes), NO diagnosis
            y_test = st.session_state["y_test"]    # pd.Series: shape (N_test,), ground truth
            test_df = st.session_state["test_df"]  # pd.DataFrame: complete 20% table with labels

            # All partition metadata:
            split_data = st.session_state["split_data"]
            classes = split_data["classes"]        # List of cancer diagnoses
            gene_cols = split_data["gene_cols"]    # List of gene names
            ```

            ---

            ##### 2. Accessing Programmatically via Python (Inside `src/models.py` or `src/evaluation.py`)
            Teammates writing backend modules or unit tests can import and execute the split directly:
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

            ---

            ##### 3. Model Training Contract (Step 3 Teammate)
            All 3 models must be fitted strictly on `(X_train, y_train)`:
            ```python
            mlp_model.fit(X_train, y_train)
            rf_model.fit(X_train, y_train)
            lr_model.fit(X_train, y_train)
            ```

            ##### 4. Evaluation Contract (Step 4 Teammate)
            Generate predictions using `X_test` (diagnosis hidden), then benchmark against `y_test`:
            ```python
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
            # Evaluate: accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred), roc_curve(...)
            ```
            """
        )

    st.markdown("---")
    st.markdown("#### Proceed to Step 3: Model Training")
    st.info(
        "80/20 train/test split successfully established and verified. "
        "The partitions are now staged in `st.session_state` and ready for Step 3 "
        "(Neural Network MLP, Random Forest, and Logistic Regression training)."
    )


def main() -> None:
    """Main application execution pipeline for Steps 1 and 2."""
    template_path = ensure_template_exists()

    # Track active workflow step
    if "current_step" not in st.session_state:
        st.session_state["current_step"] = 1

    render_header(current_step=st.session_state["current_step"])
    render_format_instructions(template_path)

    st.markdown("---")

    # CSV File Uploader
    uploaded_file = st.file_uploader(
        "Upload RNA-seq CSV File",
        type=["csv"],
        key="rna_seq_csv_uploader",
        help="Upload a CSV file conforming to the specification described above.",
    )

    if uploaded_file is not None:
        try:
            st.session_state["raw_df"] = load_dataset_from_csv(uploaded_file)
            st.session_state["source_label"] = f"Uploaded File: {uploaded_file.name}"
            # Reset split state when a new file is uploaded
            st.session_state["current_step"] = 1
            st.session_state["split_data"] = None
        except Exception as exc:
            st.error(f"Error reading CSV file: {exc}")
            return

    # Check if a dataset is available in session state
    if "raw_df" not in st.session_state:
        st.info("Upload a CSV file or click 'Load Built-in Sample Data' above to proceed with exploratory visualization.")
        return

    raw_df: pd.DataFrame = st.session_state["raw_df"]
    source_label: str = st.session_state.get("source_label", "Active Dataset")

    # Validate dataset structure
    is_valid, validation_msg, metadata = validate_dataset(raw_df)

    if not is_valid or metadata is None:
        st.error(f"Dataset Validation Failed: {validation_msg}")
        return

    st.success(f"Validation Succeeded: {source_label}")

    # Dataset Summary Metrics
    st.markdown("#### Dataset Overview")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Total Patients", metadata["n_patients"])
    metric_col2.metric("Gene Features", metadata["n_genes"])
    metric_col3.metric("Diagnosis Classes", len(metadata["classes"]))

    # Class distribution breakdown
    with st.expander("View Class Distribution and Data Preview", expanded=False):
        dist_df = pd.DataFrame(
            list(metadata["class_distribution"].items()),
            columns=["Diagnosis Class", "Patient Count"],
        )
        dist_col, preview_col = st.columns([1, 2])
        with dist_col:
            st.markdown("**Sample Counts per Class:**")
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
        with preview_col:
            st.markdown("**First 5 Rows of Normalized Data:**")
            st.dataframe(metadata["clean_df"].head(5), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Exploratory Dimensionality Reduction Section
    st.markdown("### Exploratory Dimensionality Reduction (PCA & UMAP)")
    st.markdown(
        """
        <div class="info-callout">
            Clear clustering of same-colored points indicates good class separability for model training.
            Inspect the 2D Principal Component Analysis (PCA) and Uniform Manifold Approximation
            and Projection (UMAP) projections below to evaluate biological group separation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_pca, col_umap = st.columns(2)

    with col_pca:
        with st.spinner("Computing 2D PCA projection..."):
            fig_pca = plot_pca(
                features=metadata["features"],
                labels=metadata["labels"],
                patient_ids=metadata["patient_ids"],
                random_state=42,
            )
            st.plotly_chart(fig_pca, use_container_width=True)

    with col_umap:
        with st.spinner("Computing 2D UMAP projection..."):
            fig_umap = plot_umap(
                features=metadata["features"],
                labels=metadata["labels"],
                patient_ids=metadata["patient_ids"],
                random_state=42,
            )
            st.plotly_chart(fig_umap, use_container_width=True)

    st.markdown("---")

    # Proceed to Step 2 Control
    if st.session_state["current_step"] == 1:
        st.markdown("#### Proceed to Step 2: Train / Test Split")
        st.write(
            "Confirm whether the dimensionality reduction clusters exhibit sufficient class separation. "
            "Proceeding will split the verified dataset into an 80% training set and a 20% test set."
        )

        if st.button("Proceed to Step 2: Split Dataset (80/20)", key="proceed_to_step2_btn", type="primary"):
            st.session_state["dataset"] = metadata
            st.session_state["current_step"] = 2
            st.rerun()

    # If Step 2 or beyond, render Step 2 interface
    if st.session_state["current_step"] >= 2:
        render_step2_split(metadata)


if __name__ == "__main__":
    main()
