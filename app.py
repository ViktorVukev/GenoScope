"""GenoScope: Web-based bioinformatics application for classifying cancer types
from RNA-seq gene expression data.
Step 1: Data Upload, Template, and Exploratory Visualization.
Step 2: Stratified Train / Test Split (80% Training / 20% Testing).
Step 3: Model Training (MLP Neural Network, Random Forest, Logistic Regression).
Step 4: Model Evaluation, Comparison, and Single-Patient Clinical Inference.
"""

from pathlib import Path
from typing import Any, Dict
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import (
    generate_sample_dataset,
    load_dataset_from_csv,
    save_sample_template,
    split_dataset_from_metadata,
    validate_dataset,
)
from src.evaluation import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curves,
)
from src.models import predict_single_patient_df, train_all_models
from src.visualization import CATEGORICAL_PALETTE, plot_pca, plot_umap

st.set_page_config(
    page_title="GenoScope - RNA-seq Cancer Classification",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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


def preview_large_df(df: pd.DataFrame, max_rows: int = 15, max_cols: int = 12) -> pd.DataFrame:
    """Return a lightweight slice of a wide DataFrame so Streamlit UI stays responsive."""
    if df.shape[1] <= max_cols:
        return df.head(max_rows)
    head_cols = list(df.columns[: max_cols - 1]) + [df.columns[-1]]
    return df[head_cols].head(max_rows)


def ensure_template_exists() -> Path:
    """Ensure the template CSV file exists on disk and return its path."""
    template_path = Path("data/sample_template.csv")
    if not template_path.exists():
        save_sample_template(template_path)
    return template_path


def render_header(current_step: int = 1) -> None:
    """Render top application header and workflow pipeline summary."""
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
    """Render data specification instructions, visual table preview, and download controls."""
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
            st.session_state["model_results"] = None
            st.session_state["fig_pca"] = None
            st.session_state["fig_umap"] = None


def render_step2_split(metadata: Dict[str, Any]) -> None:
    """Render Step 2: Stratified Train / Test Split and developer integration tables."""
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
            st.session_state["X_train"] = split_data["X_train"]
            st.session_state["y_train"] = split_data["y_train"]
            st.session_state["X_test"] = split_data["X_test"]
            st.session_state["y_test"] = split_data["y_test"]
            st.session_state["train_df"] = split_data["train_df"]
            st.session_state["test_df"] = split_data["test_df"]
            st.session_state["test_features_df"] = split_data["test_features_df"]
            st.session_state["test_labels_df"] = split_data["test_labels_df"]
            st.session_state["model_results"] = None
        except Exception as exc:
            st.error(f"Error computing stratified split: {exc}")
            return
    else:
        split_data = st.session_state["split_data"]

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    train_pct = (1.0 - split_data["test_ratio"]) * 100.0
    test_pct = split_data["test_ratio"] * 100.0
    m_col1.metric("Training Samples", f"{split_data['train_size']} patients", f"{train_pct:.0f}% of total")
    m_col2.metric("Testing Samples", f"{split_data['test_size']} patients", f"{test_pct:.0f}% of total")
    m_col3.metric("Gene Features", f"{len(split_data['gene_cols'])} genes")
    m_col4.metric("Diagnosis Classes", f"{len(split_data['classes'])} classes")

    st.markdown("#### Stratification Verification")
    st.caption("Verification that relative class frequencies are identically maintained across both partitions:")

    balance_rows = []
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

    st.dataframe(pd.DataFrame(balance_rows), use_container_width=True, hide_index=True)

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

    st.markdown("#### Inspect Partitioned Tables (Preview)")
    is_large = len(split_data["gene_cols"]) > 50
    if is_large:
        st.caption(
            f"Showing first 15 rows and a subset of columns out of {len(split_data['gene_cols'])} genes "
            "for fast browser rendering."
        )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Training Set (80% with Labels)",
        "Test Features (20% Target Hidden)",
        "Test Ground Truth (20% for Evaluation)",
        "Full Test Set (20% Complete)",
    ])

    with tab1:
        st.dataframe(preview_large_df(split_data["train_df"]), use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(preview_large_df(split_data["test_features_df"]), use_container_width=True, hide_index=True)
    with tab3:
        st.dataframe(split_data["test_labels_df"].head(30), use_container_width=True, hide_index=True)
    with tab4:
        st.dataframe(preview_large_df(split_data["test_df"]), use_container_width=True, hide_index=True)


def render_single_patient_upload() -> None:
    """Render Single-Patient CSV File Upload and Demo File Generator for clinical inference."""
    st.markdown("---")
    st.markdown("### Single-Patient Clinical Diagnosis (Upload Patient CSV)")
    st.markdown(
        """
        <div class="info-callout">
            Upload a CSV file containing a <b>single patient's gene expression profile</b> to predict their
            cancer type across all three trained models. If the CSV includes the ground-truth diagnosis column,
            the application will automatically verify whether each model's prediction matches the real diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    split_data = st.session_state["split_data"]
    model_results = st.session_state["model_results"]
    test_df: pd.DataFrame = split_data["test_df"]
    test_features_df: pd.DataFrame = split_data["test_features_df"]
    patient_col: str = split_data["patient_col"]
    target_col: str = split_data["target_col"]
    gene_cols = split_data["gene_cols"]

    # Helper expander to export 1-patient demo CSV files from the last rows of the 20% test split
    with st.expander("Export Demo Single-Patient CSV Files (From Last Rows of 20% Test Split)", expanded=False):
        st.write(
            "Select any patient from the last 10 rows of the held-out 20% test split to download a ready-to-upload "
            "single-patient CSV file for live testing and demonstration."
        )
        tail_df = test_df.tail(min(10, len(test_df)))
        tail_indices = list(tail_df.index)
        tail_labels = [
            f"Row {idx + 1}: {tail_df.loc[idx, patient_col]} (True Diagnosis: {tail_df.loc[idx, target_col]})"
            for idx in tail_indices
        ]

        selected_tail_label = st.selectbox(
            "Choose a Held-Out Test Patient to Export",
            options=tail_labels,
            key="export_demo_patient_select",
        )
        chosen_idx = tail_indices[tail_labels.index(selected_tail_label)]
        chosen_patient_id = str(test_df.loc[chosen_idx, patient_col])

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            single_with_label_csv = test_df.loc[[chosen_idx]].to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"Download {chosen_patient_id} CSV (With True Diagnosis)",
                data=single_with_label_csv,
                file_name=f"patient_{chosen_patient_id}_with_diagnosis.csv",
                mime="text/csv",
                key="dl_single_with_label_btn",
            )
        with dl_col2:
            single_blind_csv = test_features_df.loc[[chosen_idx]].to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"Download {chosen_patient_id} CSV (Blind - No Diagnosis)",
                data=single_blind_csv,
                file_name=f"patient_{chosen_patient_id}_blind.csv",
                mime="text/csv",
                key="dl_single_blind_btn",
            )

    # Single-Patient File Uploader
    single_file = st.file_uploader(
        "Upload Single-Patient CSV File (1 Patient Row + Gene Expressions)",
        type=["csv"],
        key="single_patient_csv_uploader",
    )

    if single_file is None:
        return

    try:
        uploaded_patient_df = load_dataset_from_csv(single_file)
        inference_result = predict_single_patient_df(
            patient_df=uploaded_patient_df,
            model_results=model_results,
            gene_cols=gene_cols,
            patient_col=patient_col,
            target_col=target_col,
        )
    except Exception as exc:
        st.error(f"Single-Patient Prediction Error: {exc}")
        return

    patient_id = inference_result["patient_id"]
    true_diag = inference_result["true_diagnosis"]
    preds = inference_result["predictions"]

    st.markdown("#### Clinical Prediction Report")
    if true_diag is not None:
        st.markdown(
            f"**Patient ID:** `{patient_id}` &nbsp;|&nbsp; **Actual Ground-Truth Diagnosis:** `{true_diag}`"
        )
    else:
        st.markdown(
            f"**Patient ID:** `{patient_id}` &nbsp;|&nbsp; **Ground-Truth Diagnosis:** *Not provided (Blind Inference)*"
        )

    pred_cols = st.columns(3)
    for idx, key in enumerate(["mlp", "rf", "lr"]):
        model_pred = preds[key]
        with pred_cols[idx]:
            st.metric(
                label=model_pred["model_name"],
                value=model_pred["predicted_class"],
                delta=f"{model_pred['confidence_pct']:.1f}% confidence",
            )
            if model_pred["is_correct"] is True:
                st.success(f"Correct: Matches real diagnosis ({true_diag})")
            elif model_pred["is_correct"] is False:
                st.error(f"Incorrect: Real diagnosis is {true_diag}")
            else:
                st.info("Blind prediction (no ground-truth column in file)")

    # Side-by-side Probability Distribution Chart across all 3 models
    fig_probs = go.Figure()
    classes = split_data["classes"]
    for idx, key in enumerate(["mlp", "rf", "lr"]):
        model_pred = preds[key]
        probs_pct = [model_pred["probabilities"].get(c, 0.0) * 100.0 for c in classes]
        fig_probs.add_trace(
            go.Bar(
                name=model_pred["model_name"],
                x=classes,
                y=probs_pct,
                marker_color=CATEGORICAL_PALETTE[idx % len(CATEGORICAL_PALETTE)],
                text=[f"{p:.1f}%" for p in probs_pct],
                textposition="auto",
            )
        )

    fig_probs.update_layout(
        title=f"Predicted Class Probability Distribution for Patient {patient_id}",
        xaxis_title="Cancer Diagnosis Class",
        yaxis_title="Predicted Probability (%)",
        yaxis=dict(range=[0, 105], gridcolor="#E2E8F0"),
        barmode="group",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A", family="sans-serif"),
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig_probs, use_container_width=True)


def render_step3_and_step4() -> None:
    """Render Step 3 (Model Training) and Step 4 (Evaluation & Comparison)."""
    st.markdown("---")
    st.markdown("### Step 3: Model Training (Deep Learning vs. Classical Baselines)")
    st.markdown(
        """
        <div class="info-callout">
            Train three distinct classification models on the exact same <b>80% training split</b>:
            a <b>Multi-Layer Perceptron (MLP) Neural Network</b>, a <b>Random Forest</b> ensemble,
            and multinomial <b>Logistic Regression</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns(4)
    with cfg_col1:
        mlp_arch_label = st.selectbox(
            "MLP Hidden Layers",
            options=["(128, 64)", "(256, 128, 64)", "(64, 32)"],
            index=0,
            key="mlp_arch_select",
        )
        arch_map = {
            "(128, 64)": (128, 64),
            "(256, 128, 64)": (256, 128, 64),
            "(64, 32)": (64, 32),
        }
        mlp_layers = arch_map[mlp_arch_label]

    with cfg_col2:
        rf_trees = st.number_input(
            "Random Forest Trees",
            min_value=50,
            max_value=500,
            value=200,
            step=50,
            key="rf_trees_input",
        )

    with cfg_col3:
        lr_c = st.number_input(
            "Logistic Regression (C)",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.25,
            key="lr_c_input",
        )

    with cfg_col4:
        top_n_genes = st.slider(
            "Top Biomarker Genes",
            min_value=5,
            max_value=30,
            value=15,
            step=5,
            key="top_n_genes_slider",
        )

    if st.button("Train All 3 Models & Run Evaluation", type="primary", key="train_models_btn"):
        X_train = st.session_state["X_train"]
        y_train = st.session_state["y_train"]
        X_test = st.session_state["X_test"]
        split_data = st.session_state["split_data"]

        with st.spinner("Training Neural Network (MLP), Random Forest, and Logistic Regression on 80% split..."):
            model_results = train_all_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                classes=split_data["classes"],
                mlp_hidden_layers=mlp_layers,
                rf_n_estimators=int(rf_trees),
                lr_c=float(lr_c),
                random_state=int(split_data["random_state"]),
            )
            st.session_state["model_results"] = model_results
            st.session_state["current_step"] = 4

    if "model_results" not in st.session_state or st.session_state["model_results"] is None:
        return

    model_results = st.session_state["model_results"]
    y_test = st.session_state["y_test"]
    split_data = st.session_state["split_data"]
    classes = split_data["classes"]
    gene_cols = split_data["gene_cols"]

    st.markdown("---")
    st.markdown("### Step 4: Model Evaluation, Comparison & Biological Interpretability")

    summary_rows = []
    metric_cols = st.columns(3)

    for idx, (key, res) in enumerate(model_results.items()):
        metrics = calculate_metrics(y_test, res.y_pred)
        with metric_cols[idx]:
            st.metric(
                label=res.model_name,
                value=f"{metrics['accuracy_pct']:.1f}% Accuracy",
                delta=f"{metrics['correct_count']} / {metrics['total_count']} patients ({res.training_time_sec:.2f}s)",
            )
        summary_rows.append({
            "Model": res.model_name,
            "Test Accuracy": metrics["formatted_accuracy"],
            "Correct Predictions": metrics["correct_count"],
            "Total Test Patients": metrics["total_count"],
            "Training Time (s)": f"{res.training_time_sec:.2f}s",
        })

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    model_tabs = st.tabs([
        "Neural Network (MLP)",
        "Random Forest",
        "Logistic Regression",
    ])

    for tab, key in zip(model_tabs, ["mlp", "rf", "lr"]):
        res = model_results[key]
        metrics = calculate_metrics(y_test, res.y_pred)

        with tab:
            st.markdown(
                f"""
                <div class="success-callout">
                    <b>{res.model_name} Performance:</b> {metrics['formatted_accuracy']}
                </div>
                """,
                unsafe_allow_html=True,
            )

            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                fig_cm = plot_confusion_matrix(
                    y_true=y_test,
                    y_pred=res.y_pred,
                    classes=classes,
                    model_name=res.model_name,
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            with chart_col2:
                fig_roc = plot_roc_curves(
                    y_true=y_test,
                    y_prob=res.y_prob,
                    classes=classes,
                    model_name=res.model_name,
                )
                st.plotly_chart(fig_roc, use_container_width=True)

            fig_imp = plot_feature_importance(
                model=res.estimator,
                gene_cols=gene_cols,
                top_n=top_n_genes,
                model_name=res.model_name,
            )
            if fig_imp is not None:
                st.plotly_chart(fig_imp, use_container_width=True)

    # Render the Single-Patient CSV Upload & Clinical Diagnosis section
    render_single_patient_upload()


def main() -> None:
    """Main application execution pipeline for Steps 1, 2, 3, and 4."""
    template_path = ensure_template_exists()

    if "current_step" not in st.session_state:
        st.session_state["current_step"] = 1

    render_header(current_step=st.session_state["current_step"])
    render_format_instructions(template_path)

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload RNA-seq CSV File",
        type=["csv"],
        key="rna_seq_csv_uploader",
        help="Upload a CSV file conforming to the specification described above.",
    )

    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("uploaded_file_id") != file_id:
            try:
                with st.spinner(f"Loading {uploaded_file.name}..."):
                    st.session_state["raw_df"] = load_dataset_from_csv(uploaded_file)
                st.session_state["uploaded_file_id"] = file_id
                st.session_state["source_label"] = f"Uploaded File: {uploaded_file.name}"
                st.session_state["current_step"] = 1
                st.session_state["split_data"] = None
                st.session_state["model_results"] = None
                st.session_state["fig_pca"] = None
                st.session_state["fig_umap"] = None
            except Exception as exc:
                st.error(f"Error reading CSV file: {exc}")
                return

    if "raw_df" not in st.session_state:
        st.info("Upload a CSV file or click 'Load Built-in Sample Data' above to proceed with exploratory visualization.")
        return

    raw_df: pd.DataFrame = st.session_state["raw_df"]
    source_label: str = st.session_state.get("source_label", "Active Dataset")

    if "validated_metadata" not in st.session_state or st.session_state.get("validated_source") != source_label:
        with st.spinner("Validating dataset structure..."):
            is_valid, validation_msg, metadata = validate_dataset(raw_df)
        if not is_valid or metadata is None:
            st.error(f"Dataset Validation Failed: {validation_msg}")
            return
        st.session_state["validated_metadata"] = metadata
        st.session_state["validated_source"] = source_label
    else:
        metadata = st.session_state["validated_metadata"]

    st.success(f"Validation Succeeded: {source_label}")

    st.markdown("#### Dataset Overview")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Total Patients", metadata["n_patients"])
    metric_col2.metric("Gene Features", metadata["n_genes"])
    metric_col3.metric("Diagnosis Classes", len(metadata["classes"]))

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
            st.markdown("**First 5 Rows of Normalized Data (Preview):**")
            st.dataframe(preview_large_df(metadata["clean_df"], max_rows=5), use_container_width=True, hide_index=True)

    st.markdown("---")

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
        if st.session_state.get("fig_pca") is None:
            with st.spinner("Computing 2D PCA projection..."):
                st.session_state["fig_pca"] = plot_pca(
                    features=metadata["features"],
                    labels=metadata["labels"],
                    patient_ids=metadata["patient_ids"],
                    random_state=42,
                )
        st.plotly_chart(st.session_state["fig_pca"], use_container_width=True)

    with col_umap:
        if st.session_state.get("fig_umap") is None:
            with st.spinner("Computing 2D UMAP projection..."):
                st.session_state["fig_umap"] = plot_umap(
                    features=metadata["features"],
                    labels=metadata["labels"],
                    patient_ids=metadata["patient_ids"],
                    random_state=42,
                )
        st.plotly_chart(st.session_state["fig_umap"], use_container_width=True)

    st.markdown("---")

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

    if st.session_state["current_step"] >= 2:
        render_step2_split(metadata)
        render_step3_and_step4()


if __name__ == "__main__":
    main()