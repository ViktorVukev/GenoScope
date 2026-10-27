"""GenoScope: Web-based bioinformatics application for classifying cancer types
from RNA-seq gene expression data.
Step 1: Data Upload, Template, and Exploratory Visualization.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import streamlit as st

from src.data_loader import (
    generate_sample_dataset,
    load_dataset_from_csv,
    save_sample_template,
    validate_dataset,
)
from src.visualization import plot_pca, plot_umap

# Configure page layout and metadata
st.set_page_config(
    page_title="GenoScope - RNA-seq Cancer Classification",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Clinical custom styling
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


def render_header() -> None:
    """Render top application header and workflow pipeline summary."""
    st.markdown(
        """
        <div class="header-box">
            <span class="step-badge">Stage 1 of 4: Exploratory Analysis</span>
            <div class="header-title">GenoScope: RNA-seq Cancer Classification and Biological Interpretability</div>
            <div class="header-subtitle">
                Compare Deep Learning (Multi-Layer Perceptron) with classical machine learning
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
        if st.button("Load Built-in Sample Data", key="load_sample_data_btn", help="Quickly test the app using the template dataset."):
            st.session_state["raw_df"] = load_dataset_from_csv(template_path)
            st.session_state["source_label"] = "Built-in Sample Template (30 patients, 8 genes, 3 classes)"
            st.session_state["ready_for_step2"] = False


def main() -> None:
    """Main application execution pipeline for Step 1."""
    template_path = ensure_template_exists()
    render_header()
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
            st.session_state["ready_for_step2"] = False
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
    st.markdown("#### Proceed to Model Training")
    st.write(
        "Confirm whether the dimensionality reduction clusters exhibit sufficient class separation. "
        "Proceeding will split the verified dataset into an 80% training set and a 20% test set."
    )

    if st.button("Proceed to Model Training (80/20 Split)", key="proceed_to_step2_btn", type="primary"):
        st.session_state["dataset"] = metadata
        st.session_state["ready_for_step2"] = True
        st.session_state["current_step"] = 2

    if st.session_state.get("ready_for_step2", False):
        st.info(
            "Dataset successfully staged for Step 2 (Train / Test Split and Model Training). "
            "The 80% training split and 20% held-out test split will be prepared in the next stage."
        )


if __name__ == "__main__":
    main()
