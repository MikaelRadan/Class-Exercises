import io

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# -------------------------------------------------------------------------
# 1. Page Configuration & Emerald-Gold Glassmorphism Theme
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Corporate Bankruptcy Risk Intelligence",
    page_icon="🏛️",
    layout="wide",
)

st.markdown("""
<style>
    .stApp {
        background-image:
            radial-gradient(circle at 10% 20%, rgba(176, 141, 87, 0.22) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(62, 90, 63, 0.30) 0%, transparent 45%),
            linear-gradient(135deg, rgba(11, 15, 13, 0.90) 0%, rgba(31, 42, 31, 0.88) 100%),
            url("https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?auto=format&fit=crop&w=2000&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #D9C7A3;
    }
    section[data-testid="stSidebar"] {
        background: rgba(11, 15, 13, 0.84) !important;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border-right: 1px solid rgba(176, 141, 87, 0.28);
    }
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #D9C7A3 0%, #B08D57 60%, #8CA98C 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #8CA98C;
        margin-bottom: 1.5rem;
        letter-spacing: 0.4px;
    }
    .glass-card {
        background: rgba(31, 42, 31, 0.60);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(140, 169, 140, 0.25);
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 18px;
        box-shadow: 0 12px 32px 0 rgba(0, 0, 0, 0.45);
    }
    .card-header {
        color: #D9C7A3;
        font-size: 1.15rem;
        font-weight: 700;
        border-bottom: 1px solid rgba(176, 141, 87, 0.35);
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #B08D57 0%, #8E6D3B 100%) !important;
        color: #0B0F0D !important;
        font-weight: 800 !important;
        font-size: 1.05rem !important;
        border: 1px solid #D9C7A3 !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.4rem !important;
        box-shadow: 0 6px 20px rgba(176, 141, 87, 0.35) !important;
        transition: all 0.25s ease-in-out !important;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(176, 141, 87, 0.55) !important;
        background: linear-gradient(135deg, #D9C7A3 0%, #B08D57 100%) !important;
    }
    .risk-badge-safe {
        background: rgba(62, 90, 63, 0.65);
        backdrop-filter: blur(12px);
        color: #D9C7A3;
        border: 1px solid rgba(140, 169, 140, 0.4);
        border-left: 6px solid #8CA98C;
        padding: 18px 22px;
        border-radius: 12px;
        font-size: 1.02rem;
        line-height: 1.6;
    }
    .risk-badge-danger {
        background: rgba(90, 28, 28, 0.70);
        backdrop-filter: blur(12px);
        color: #FDE8E8;
        border: 1px solid rgba(239, 68, 68, 0.45);
        border-left: 6px solid #B08D57;
        padding: 18px 22px;
        border-radius: 12px;
        font-size: 1.02rem;
        line-height: 1.6;
    }
    label, .stMarkdown p {
        color: #D9C7A3 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #B08D57 !important;
        font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🏛️ Corporate Bankruptcy Risk Intelligence Platform</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Automated ML Pipeline Deployment • Preprocessing, Scaling & '
    'Random Forest Risk Scoring</div>',
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------------
# 2. Cached Model & Reference Dataset Loading
# -------------------------------------------------------------------------
KEY_COLS = [
    'Net Income to Total Assets',
    'Retained Earnings to Total Assets',
    'Continuous interest rate (after tax)',
    'Debt ratio %',
    'Borrowing dependency',
    'ROA(A) before interest and % after tax',
]


@st.cache_resource
def load_pipeline():
    """Load the serialized Scikit-Learn pipeline."""
    return joblib.load("taiwan_bankruptcy_pipeline.joblib")


@st.cache_data
def load_reference_data():
    """Load and clean the reference dataset (same cleaning as the notebook)."""
    df = pd.read_csv("Taiwanese Bankruptcy.csv")
    df.columns = df.columns.str.strip()
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)
    X = df.drop(columns=['Bankrupt?'])
    y = df['Bankrupt?']
    return X, y


try:
    pipeline = load_pipeline()
    X_ref, y_ref = load_reference_data()
except Exception as e:
    st.error(f"**Deployment Error:** Unable to load model or dataset files. Details: `{e}`")
    st.stop()

X_MEDIAN = X_ref.median()


def preprocess_uploaded_dataframe(raw_df, reference_X):
    """
    Align an uploaded DataFrame with the training features.
    Returns (aligned_X, missing_cols). Missing columns / NaNs are filled with
    the reference median, and the list of missing columns is returned so the
    UI can warn the user.
    """
    clean_df = raw_df.copy()
    clean_df.columns = clean_df.columns.str.strip()

    id_values = None
    for candidate in ['Company', 'Company Name', 'Firm', 'ID', 'Name']:
        if candidate in clean_df.columns:
            id_values = clean_df[candidate].astype(str).tolist()
            break

    missing_cols = []
    aligned_X = pd.DataFrame(index=clean_df.index)
    for col in reference_X.columns:
        if col in clean_df.columns:
            aligned_X[col] = pd.to_numeric(clean_df[col], errors='coerce').fillna(
                reference_X[col].median()
            )
        else:
            aligned_X[col] = reference_X[col].median()
            missing_cols.append(col)

    if id_values is None:
        id_values = [f"Company #{i + 1}" for i in range(len(aligned_X))]
    aligned_X.index = id_values
    aligned_X.index.name = "Company"
    return aligned_X, missing_cols


@st.cache_data
def generate_sample_csv():
    """3-company sample: 1 healthy + 2 bankrupt rows, labeled by the REAL dataset label."""
    healthy_idx = y_ref[y_ref == 0].index[0]
    bankrupt_idx = list(y_ref[y_ref == 1].index[:2])
    idx = [healthy_idx] + bankrupt_idx
    sample_df = X_ref.loc[idx].copy()
    names = [
        f"Company A (Actual: {'Bankrupt' if y_ref.loc[i] == 1 else 'Healthy'})"
        for i in idx
    ]
    sample_df.insert(0, "Company Name", names)
    return sample_df.to_csv(index=False).encode('utf-8')


# -------------------------------------------------------------------------
# 3. Sidebar
# -------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Policy Calibration")
    threshold = st.slider(
        "Audit Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.05,
        help="Lower threshold = catch more distressed firms (higher Recall) "
             "but more false alarms.",
    )
    st.caption("ℹ️ Default `0.40` prioritizes catching distressed firms (high Recall) "
               "over minimizing false alarms.")
    st.markdown("---")

    st.markdown("### 📂 Evaluation Mode")
    input_mode = st.radio(
        "Select Data Source:",
        [
            "📥 Batch File Upload & Multi-Firm Audit (CSV)",
            "🎛️ Scenario Stress Testing (Sliders)",
            "🏢 Historical Company Audit (Dataset)",
        ],
    )
    st.markdown("---")
    st.caption("🎨 **Theme:** Emerald & Antique Gold")

# -------------------------------------------------------------------------
# 4. Main Tabs
# -------------------------------------------------------------------------
tab_eval, tab_specs = st.tabs(["📊 Risk Assessment Engine", "📖 Pipeline Specs & Disclaimer"])

with tab_eval:
    # =====================================================================
    # MODE 1: BATCH CSV
    # =====================================================================
    if input_mode == "📥 Batch File Upload & Multi-Firm Audit (CSV)":
        st.markdown(
            '<div class="glass-card">'
            '<div class="card-header">📥 Automated Multi-Company Pipeline Processing</div>'
            'Upload a CSV with financial ratios for multiple companies. The app strips column '
            'whitespace, aligns the 94 accounting features, applies <code>StandardScaler</code> '
            'and computes a bankruptcy <b>risk score</b> using the tuned '
            '<code>RandomForestClassifier</code>.'
            '</div>',
            unsafe_allow_html=True,
        )

        up_col1, up_col2 = st.columns([2, 1])
        with up_col1:
            uploaded_file = st.file_uploader(
                "📂 Upload Company Financial Statements (CSV)", type=["csv"]
            )
        with up_col2:
            st.markdown("##### 🧪 Need a Test File?")
            st.download_button(
                label="⬇️ Download Sample 3-Companies CSV",
                data=generate_sample_csv(),
                file_name="sample_3_companies_taiwan_format.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption("Sample rows come from the original dataset (labels are the real ones). "
                       "They may have been part of the training data, so results do not "
                       "reflect true out-of-sample performance.")

        if uploaded_file is not None:
            try:
                raw_user_df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Could not read the CSV file: `{e}`")
                st.stop()
            aligned_batch_X, missing_cols = preprocess_uploaded_dataframe(raw_user_df, X_ref)
            st.success(
                f"✅ File loaded: **{len(aligned_batch_X)} companies** aligned to the "
                f"{X_ref.shape[1]}-feature pipeline."
            )
            if missing_cols:
                st.warning(
                    f"⚠️ **{len(missing_cols)} of {X_ref.shape[1]} required features were not "
                    f"found** in your file and were filled with dataset medians. "
                    f"Results may be unreliable."
                )
                with st.expander("Show missing features"):
                    st.write(missing_cols)
        else:
            st.info("💡 **No CSV uploaded:** showing a default 3-company sample. "
                    "You can edit cells directly or upload your own file.")
            raw_default_df = pd.read_csv(io.BytesIO(generate_sample_csv()))
            aligned_batch_X, _ = preprocess_uploaded_dataframe(raw_default_df, X_ref)

        st.markdown("##### ✏️ Edit key ratios (double-click a cell). "
                    "You can also add or delete rows:")
        editor_df = aligned_batch_X[KEY_COLS].reset_index()  # 'Company' becomes a column
        edited_df = st.data_editor(
            editor_df,
            use_container_width=True,
            num_rows="dynamic",
            key=f"editor_{'up' if uploaded_file is not None else 'sample'}",
        )

        # Build the full feature matrix from the edited table
        full_lookup = aligned_batch_X.reset_index().drop_duplicates("Company").set_index("Company")
        edited_clean = edited_df.dropna(subset=KEY_COLS).reset_index(drop=True)
        dropped = len(edited_df) - len(edited_clean)
        if dropped > 0:
            st.caption(f"ℹ️ {dropped} row(s) with empty values were ignored.")

        rows, names = [], []
        for i, r in edited_clean.iterrows():
            name = str(r["Company"]).strip() if pd.notna(r["Company"]) else ""
            if not name or name == "nan":
                name = f"Company #{i + 1}"
            if name in full_lookup.index:
                base = full_lookup.loc[name].copy()
            else:
                base = X_MEDIAN.copy()
            for c in KEY_COLS:
                base[c] = r[c]
            rows.append(base)
            names.append(name)

        if st.button("🚀 Run Automated Pipeline on All Companies", type="primary",
                     use_container_width=True):
            if not rows:
                st.error("No valid rows to score. Please fill in all key ratios.")
            else:
                final_batch_X = pd.DataFrame(rows)[X_ref.columns].reset_index(drop=True)
                scores = pipeline.predict_proba(final_batch_X)[:, 1]
                preds = (scores >= threshold).astype(int)

                total_firms = len(final_batch_X)
                high_risk_count = int(preds.sum())
                healthy_count = total_firms - high_risk_count

                st.markdown(
                    '<div class="glass-card"><div class="card-header">'
                    '📊 Portfolio Audit Results</div>',
                    unsafe_allow_html=True,
                )
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total Companies", f"{total_firms}")
                k2.metric("Flagged High-Risk (Class 1)", f"{high_risk_count}",
                          delta="Requires Audit" if high_risk_count > 0 else "Clear",
                          delta_color="inverse")
                k3.metric("Healthy (Class 0)", f"{healthy_count}")
                k4.metric("Decision Threshold", f"{threshold:.0%}")

                results_table = pd.DataFrame({
                    "Company": names,
                    "Risk Score": scores,
                    "Predicted Class": [f"Class {p}" for p in preds],
                    "Audit Recommendation": [
                        "🚨 HIGH RISK: Escalate for Credit Audit" if p == 1
                        else "✅ HEALTHY: Standard Monitoring"
                        for p in preds
                    ],
                })
                results_table = pd.concat(
                    [results_table, edited_clean[KEY_COLS].reset_index(drop=True)], axis=1
                )

                st.dataframe(
                    results_table,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Risk Score": st.column_config.ProgressColumn(
                            "Risk Score",
                            help="Model score between 0 and 1 (not a calibrated probability).",
                            min_value=0.0,
                            max_value=1.0,
                            format="%.3f",
                        )
                    },
                )

                st.download_button(
                    label="📥 Download Full Audit Report (CSV)",
                    data=results_table.to_csv(index=False).encode('utf-8'),
                    file_name="bankruptcy_audit_predictions.csv",
                    mime="text/csv",
                )
                st.markdown('</div>', unsafe_allow_html=True)

    # =====================================================================
    # MODE 2: SLIDERS
    # =====================================================================
    elif input_mode == "🎛️ Scenario Stress Testing (Sliders)":
        input_data = X_MEDIAN.to_frame().T

        def get_slider_bounds(col_name, pad_if_tight=0.05):
            q01 = float(X_ref[col_name].quantile(0.01))
            q99 = float(X_ref[col_name].quantile(0.99))
            med = float(X_ref[col_name].median())
            if (q99 - q01) < 0.005:
                min_v = max(0.0, round(med - pad_if_tight, 4))
                max_v = min(1.0, round(med + pad_if_tight, 4))
                step_v = 0.0001
            else:
                min_v = round(q01, 4)
                max_v = round(q99, 4)
                step_v = 0.0010
            med = min(max(round(med, 4), min_v), max_v)
            return min_v, max_v, med, step_v

        st.markdown(
            '<div class="glass-card">'
            '<div class="card-header">🎛️ Interactive Financial Ratio Stress Testing</div>'
            'Adjust the 6 key financial drivers (from Random Forest feature importance and '
            'correlation analysis). All remaining ratios are held at their dataset median.'
            '</div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="glass-card"><div class="card-header">'
                        '📈 Profitability & Earnings</div>', unsafe_allow_html=True)
            for feat, label in [
                ('Net Income to Total Assets', 'Net Income to Total Assets'),
                ('Retained Earnings to Total Assets', 'Retained Earnings to Total Assets'),
                ('Continuous interest rate (after tax)', 'Continuous Interest Rate (After Tax)'),
            ]:
                mn, mx, md, stp = get_slider_bounds(feat)
                input_data[feat] = st.slider(label, min_value=mn, max_value=mx,
                                             value=md, step=stp, format="%.4f")
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="glass-card"><div class="card-header">'
                        '⚠️ Debt Exposure & Return Efficiency</div>', unsafe_allow_html=True)
            for feat, label in [
                ('Debt ratio %', 'Debt Ratio % (Total Liabilities / Total Assets)'),
                ('Borrowing dependency', 'Borrowing Dependency Ratio'),
                ('ROA(A) before interest and % after tax', 'ROA(A) Before Interest & % After Tax'),
            ]:
                mn, mx, md, stp = get_slider_bounds(feat)
                input_data[feat] = st.slider(label, min_value=mn, max_value=mx,
                                             value=md, step=stp, format="%.4f")
            st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🚀 Run Bankruptcy Risk Assessment", type="primary",
                     use_container_width=True):
            score = float(pipeline.predict_proba(input_data[X_ref.columns])[0, 1])
            is_bankrupt = score >= threshold

            st.markdown('<div class="glass-card"><div class="card-header">'
                        '📊 Risk Assessment & Audit Verdict</div>', unsafe_allow_html=True)
            res_left, res_right = st.columns([1, 2])
            with res_left:
                st.metric(
                    label="Bankruptcy Risk Score",
                    value=f"{score:.2%}",
                    delta=f"{score - threshold:+.2%} vs. Threshold ({threshold:.0%})",
                    delta_color="inverse",
                )
                st.progress(min(max(score, 0.0), 1.0))
            with res_right:
                if is_bankrupt:
                    st.markdown(f"""
                    <div class="risk-badge-danger">
                        🚨 <b>HIGH DISTRESS RISK — FLAGGED AS CLASS 1</b><br>
                        The model's risk score (<b>{score:.2%}</b>) meets or exceeds the audit
                        threshold (<b>{threshold:.0%}</b>).<br>
                        <b>Recommended Action:</b> Escalate this firm for credit committee audit.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="risk-badge-safe">
                        ✅ <b>LOW DISTRESS RISK — CLASSIFIED AS CLASS 0</b><br>
                        The model's risk score (<b>{score:.2%}</b>) is below the audit
                        threshold (<b>{threshold:.0%}</b>).<br>
                        <b>Recommended Action:</b> Standard periodic financial monitoring.
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # =====================================================================
    # MODE 3: HISTORICAL INSPECTOR
    # =====================================================================
    else:
        st.markdown('<div class="glass-card"><div class="card-header">'
                    '🏢 Historical Company Record Inspector</div>', unsafe_allow_html=True)
        idx = st.number_input(
            f"Company Row Index (0 to {len(X_ref) - 1}) — try 1212 for a distressed case:",
            min_value=0,
            max_value=len(X_ref) - 1,
            value=min(1212, len(X_ref) - 1),
            step=1,
        )
        input_data = X_ref.iloc[[idx]].copy()
        ground_truth = int(y_ref.iloc[idx])

        status_text = ("🔴 1 (Confirmed Historical Bankruptcy)" if ground_truth == 1
                       else "🟢 0 (Historically Healthy / Active)")
        st.info(f"**Historical Ground Truth for Company #{idx}:** {status_text}")
        st.caption("⚠️ This row may have been part of the model's training data, so a correct "
                   "prediction here is not proof of generalization.")

        with st.expander(f"📄 Inspect all {X_ref.shape[1]} financial ratios"):
            st.dataframe(input_data, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🚀 Verify Model Prediction on Selected Company", type="primary",
                     use_container_width=True):
            score = float(pipeline.predict_proba(input_data)[0, 1])
            is_bankrupt = score >= threshold
            st.metric("Model Bankruptcy Risk Score", f"{score:.2%}")
            if is_bankrupt:
                st.error(f"🚨 Flagged as **Class 1 (Bankrupt)** at threshold {threshold:.0%}")
            else:
                st.success(f"✅ Classified as **Class 0 (Healthy)** at threshold {threshold:.0%}")
            if (int(is_bankrupt)) == ground_truth:
                st.caption("✔️ Prediction matches the historical label.")
            else:
                st.caption("✖️ Prediction differs from the historical label.")

with tab_specs:
    st.markdown("""
    <div class="glass-card">
        <div class="card-header">🔬 Machine Learning Pipeline Specifications</div>
        <ul>
            <li><b>Architecture:</b> <code>Pipeline([('scaler', StandardScaler()), ('classifier', RandomForestClassifier())])</code></li>
            <li><b>Tuned Hyperparameters:</b> <code>n_estimators=200</code>, <code>max_depth=12</code>, <code>min_samples_leaf=5</code>, <code>class_weight='balanced'</code></li>
            <li><b>5-Fold CV ROC-AUC (Train):</b> <code>0.9349</code></li>
            <li><b>Held-out Test ROC-AUC:</b> <code>0.9551</code></li>
            <li><b>Test Recall at 0.40 threshold:</b> <code>79.55%</code> (35 of 44 bankruptcies caught)</li>
        </ul>
    </div>
    <div class="glass-card">
        <div class="card-header">⚠️ Important Notes & Disclaimer</div>
        <ul>
            <li>The output is a <b>risk score</b>, not a calibrated probability
                (the model uses <code>class_weight='balanced'</code> and no probability calibration).</li>
            <li>The model is trained on Taiwanese company data (1999–2009); it may not
                generalize to other markets or periods.</li>
            <li>This tool is a <b>decision-support aid only</b> and must not replace
                professional credit analysis.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
#cd "H-10/week 10/Final Exam"streamlit run App.py