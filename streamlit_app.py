"""
Streamlit frontend for the Churn Prediction system.

Features:
1. Single customer prediction with risk factors
2. Batch CSV upload with results table + download
3. Live analytics dashboard
4. Model comparison view

Run: streamlit run streamlit_app.py
Requires: FastAPI backend running on localhost:8000
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import time
from pathlib import Path

API_URL = "http://localhost:8000"

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Sidebar navigation
# ──────────────────────────────────────────────
st.sidebar.title("📊 Churn Prediction")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🔍 Single Prediction", "📁 Batch Prediction", "📈 Analytics Dashboard", "🏆 Model Comparison"],
)

# Check API health
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

api_healthy = check_api()
if api_healthy:
    st.sidebar.success("✅ API Connected")
else:
    st.sidebar.error("❌ API Offline — Start with: docker-compose up")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Stack:** scikit-learn · FastAPI · MLflow · Prometheus · Grafana · Docker"
)


# ──────────────────────────────────────────────
# Page 1: Single Prediction
# ──────────────────────────────────────────────
def page_single_prediction():
    st.title("🔍 Customer Churn Prediction")
    st.markdown("Enter customer details to predict churn probability and see risk factors.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    with col2:
        st.subheader("Services")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    with col3:
        st.subheader("Billing")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=5.0)
        total = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly * max(tenure, 1), step=50.0)

    st.markdown("---")

    if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
        if not api_healthy:
            st.error("API is not running. Start with `docker-compose up`")
            return

        payload = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": tech_support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }

        with st.spinner("Running prediction..."):
            try:
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
                result = resp.json()
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

        # Display results
        r1, r2, r3 = st.columns(3)
        prob = result["churn_probability"]
        risk = result["risk_level"]

        with r1:
            st.metric("Churn Probability", f"{prob:.1%}")
        with r2:
            st.metric("Risk Level", risk.upper())
        with r3:
            st.metric("Prediction", "⚠️ WILL CHURN" if result["will_churn"] else "✅ RETAINED")

        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Churn Risk Score"},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkred" if prob > 0.5 else "darkorange" if prob > 0.3 else "green"},
                "steps": [
                    {"range": [0, 30], "color": "#e8f5e9"},
                    {"range": [30, 50], "color": "#fff3e0"},
                    {"range": [50, 70], "color": "#fbe9e7"},
                    {"range": [70, 100], "color": "#ffebee"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": result["threshold"] * 100,
                },
            },
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch")

        # Risk factors
        if result.get("top_risk_factors"):
            st.subheader("🚩 Top Risk Factors")
            for factor in result["top_risk_factors"]:
                st.markdown(f"- **{factor}**")


# ──────────────────────────────────────────────
# Page 2: Batch Prediction
# ──────────────────────────────────────────────
def page_batch_prediction():
    st.title("📁 Batch Prediction")
    st.markdown("Upload a CSV file with customer data to get churn predictions for all rows.")

    # Expected columns info
    with st.expander("📋 Expected CSV columns (click to expand)"):
        st.markdown("Your CSV should include these columns. `customerID` and `Churn` are optional — they'll be auto-removed before prediction.")
        expected_cols = pd.DataFrame({
            "Column": [
                "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
                "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
                "MonthlyCharges", "TotalCharges",
            ],
            "Type": [
                "Male / Female", "0 or 1", "Yes / No", "Yes / No", "Integer (0-72)",
                "Yes / No", "Yes / No / No phone service",
                "DSL / Fiber optic / No",
                "Yes / No / No internet service", "Yes / No / No internet service",
                "Yes / No / No internet service", "Yes / No / No internet service",
                "Yes / No / No internet service", "Yes / No / No internet service",
                "Month-to-month / One year / Two year", "Yes / No",
                "Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic)",
                "Float", "Float",
            ],
            "Required": ["✅"] * 19,
        })
        st.dataframe(expected_cols, width="stretch", hide_index=True)

    # Sample download
    sample_path = Path("data/sample_batch.csv")
    if sample_path.exists():
        st.download_button(
            "📥 Download sample CSV (10 customers)",
            sample_path.read_text(),
            "sample_batch.csv",
            "text/csv",
        )
    else:
        st.info("💡 You can also upload the original telco_churn.csv — `customerID` and `Churn` columns will be handled automatically.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.subheader(f"Uploaded: {len(df)} customers")
        st.dataframe(df.head(), width="stretch")

        # Clean up columns if needed
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])
        actual_churn = None
        if "Churn" in df.columns:
            actual_churn = df["Churn"].copy()
            df = df.drop(columns=["Churn"])

        # Fix TotalCharges
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

        if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
            if not api_healthy:
                st.error("API is not running.")
                return

            progress = st.progress(0)
            results = []

            for i, row in df.iterrows():
                payload = row.to_dict()
                payload["SeniorCitizen"] = int(payload.get("SeniorCitizen", 0))
                payload["tenure"] = int(payload.get("tenure", 0))
                payload["MonthlyCharges"] = float(payload.get("MonthlyCharges", 0))
                payload["TotalCharges"] = float(payload.get("TotalCharges", 0))

                try:
                    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
                    r = resp.json()
                    results.append({
                        "churn_probability": r["churn_probability"],
                        "will_churn": r["will_churn"],
                        "risk_level": r["risk_level"],
                    })
                except Exception:
                    results.append({
                        "churn_probability": None,
                        "will_churn": None,
                        "risk_level": "error",
                    })

                progress.progress((i + 1) / len(df))

            results_df = pd.DataFrame(results)
            output_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

            if actual_churn is not None:
                output_df["actual_churn"] = actual_churn.values

            st.subheader("Results")

            # Summary metrics
            m1, m2, m3 = st.columns(3)
            valid = results_df.dropna()
            with m1:
                st.metric("Total Processed", len(valid))
            with m2:
                churn_count = valid["will_churn"].sum()
                st.metric("Predicted to Churn", f"{churn_count} ({churn_count/len(valid)*100:.1f}%)")
            with m3:
                avg_prob = valid["churn_probability"].mean()
                st.metric("Avg Churn Probability", f"{avg_prob:.1%}")

            # Results table
            st.dataframe(
                output_df.style.background_gradient(
                    subset=["churn_probability"], cmap="RdYlGn_r"
                ),
                width="stretch",
                height=400,
            )

            # Probability distribution
            fig = px.histogram(
                valid, x="churn_probability", nbins=20,
                title="Churn Probability Distribution",
                color_discrete_sequence=["#E24B4A"],
            )
            fig.add_vline(x=0.4453, line_dash="dash", line_color="black",
                         annotation_text="Threshold")
            st.plotly_chart(fig, width="stretch")

            # Risk level breakdown
            fig2 = px.pie(
                valid, names="risk_level", title="Risk Level Breakdown",
                color="risk_level",
                color_discrete_map={
                    "low": "#5DCAA5", "medium": "#EF9F27",
                    "high": "#E24B4A", "critical": "#993556",
                },
            )
            st.plotly_chart(fig2, width="stretch")

            # Download
            csv = output_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results CSV",
                csv,
                "churn_predictions.csv",
                "text/csv",
                use_container_width=True,
            )


# ──────────────────────────────────────────────
# Page 3: Analytics Dashboard
# ──────────────────────────────────────────────
def page_analytics():
    st.title("📈 Analytics Dashboard")
    st.markdown("Live view of the churn prediction dataset and model insights.")

    # Load the training data for analytics
    data_path = Path("data/raw/telco_churn.csv")
    if not data_path.exists():
        st.error("Dataset not found. Place telco_churn.csv in data/raw/")
        return

    df = pd.read_csv(data_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Customers", f"{len(df):,}")
    with k2:
        churn_rate = df["Churn"].mean()
        st.metric("Churn Rate", f"{churn_rate:.1%}")
    with k3:
        st.metric("Avg Monthly Charge", f"${df['MonthlyCharges'].mean():.2f}")
    with k4:
        st.metric("Avg Tenure", f"{df['tenure'].mean():.1f} months")

    st.markdown("---")

    # Row 1: Churn by Contract + Tenure distribution
    c1, c2 = st.columns(2)

    with c1:
        contract_churn = df.groupby("Contract")["Churn"].mean().reset_index()
        contract_churn.columns = ["Contract", "Churn Rate"]
        fig = px.bar(
            contract_churn, x="Contract", y="Churn Rate",
            title="Churn Rate by Contract Type",
            color="Churn Rate",
            color_continuous_scale=["#5DCAA5", "#EF9F27", "#E24B4A"],
            text_auto=".1%",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with c2:
        fig = px.histogram(
            df, x="tenure", color=df["Churn"].map({0: "Retained", 1: "Churned"}),
            title="Tenure Distribution by Churn Status",
            barmode="overlay", opacity=0.7,
            color_discrete_map={"Retained": "#5DCAA5", "Churned": "#E24B4A"},
        )
        fig.update_layout(legend_title_text="")
        st.plotly_chart(fig, width="stretch")

    # Row 2: Monthly charges + Payment method
    c3, c4 = st.columns(2)

    with c3:
        fig = px.box(
            df, x=df["Churn"].map({0: "Retained", 1: "Churned"}),
            y="MonthlyCharges",
            title="Monthly Charges by Churn Status",
            color=df["Churn"].map({0: "Retained", 1: "Churned"}),
            color_discrete_map={"Retained": "#5DCAA5", "Churned": "#E24B4A"},
        )
        fig.update_layout(showlegend=False, xaxis_title="")
        st.plotly_chart(fig, width="stretch")

    with c4:
        pay_churn = df.groupby("PaymentMethod")["Churn"].mean().reset_index()
        pay_churn.columns = ["Payment Method", "Churn Rate"]
        pay_churn = pay_churn.sort_values("Churn Rate", ascending=True)
        fig = px.bar(
            pay_churn, y="Payment Method", x="Churn Rate",
            title="Churn Rate by Payment Method",
            orientation="h",
            color="Churn Rate",
            color_continuous_scale=["#5DCAA5", "#E24B4A"],
            text_auto=".1%",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

    # Row 3: Services impact
    st.subheader("Service Impact on Churn")
    services = ["OnlineSecurity", "TechSupport", "OnlineBackup", "DeviceProtection"]
    svc_data = []
    for svc in services:
        for val in df[svc].unique():
            if val != "No internet service":
                rate = df[df[svc] == val]["Churn"].mean()
                svc_data.append({"Service": svc, "Status": val, "Churn Rate": rate})

    svc_df = pd.DataFrame(svc_data)
    fig = px.bar(
        svc_df, x="Service", y="Churn Rate", color="Status",
        barmode="group",
        color_discrete_map={"Yes": "#5DCAA5", "No": "#E24B4A"},
        title="Churn Rate: With vs Without Service",
        text_auto=".1%",
    )
    st.plotly_chart(fig, width="stretch")

    # Revenue at risk
    st.subheader("💰 Revenue at Risk")
    churned = df[df["Churn"] == 1]
    monthly_risk = churned["MonthlyCharges"].sum()
    annual_risk = monthly_risk * 12
    r1, r2 = st.columns(2)
    with r1:
        st.metric("Monthly Revenue at Risk", f"${monthly_risk:,.0f}")
    with r2:
        st.metric("Annual Revenue at Risk", f"${annual_risk:,.0f}")


# ──────────────────────────────────────────────
# Page 4: Model Comparison
# ──────────────────────────────────────────────
def page_model_comparison():
    st.title("🏆 Model Comparison")
    st.markdown("Compare all trained models side-by-side using MLflow experiment data.")

    # Try loading from MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5001")
        experiment = mlflow.get_experiment_by_name("churn-prediction")

        if experiment is None:
            st.warning("No MLflow experiment found. Run training first.")
            return

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        if runs.empty:
            st.warning("No runs found in MLflow.")
            return

        # Extract key metrics
        display_cols = {
            "tags.mlflow.runName": "Model",
            "metrics.pr_auc": "PR-AUC",
            "metrics.roc_auc": "ROC-AUC",
            "metrics.f1": "F1 Score",
            "metrics.precision_churn": "Precision",
            "metrics.recall_churn": "Recall",
            "metrics.threshold": "Threshold",
            "metrics.true_positives": "True Pos",
            "metrics.false_positives": "False Pos",
            "metrics.false_negatives": "False Neg",
        }

        available = {k: v for k, v in display_cols.items() if k in runs.columns}
        comparison = runs[list(available.keys())].rename(columns=available)

        # Highlight best model
        st.subheader("Performance Summary")
        best_idx = comparison["PR-AUC"].idxmax()
        best_model = comparison.loc[best_idx, "Model"]
        st.success(f"🏆 Best model: **{best_model}** (PR-AUC: {comparison.loc[best_idx, 'PR-AUC']:.4f})")

        # Styled table
        numeric_cols = ["PR-AUC", "ROC-AUC", "F1 Score", "Precision", "Recall"]
        present_numeric = [c for c in numeric_cols if c in comparison.columns]

        styled = comparison.style.format(
            {c: "{:.4f}" for c in present_numeric if c in comparison.columns}
        ).highlight_max(subset=present_numeric, color="#d4edda")

        st.dataframe(styled, width="stretch", hide_index=True)

        # Bar chart comparison
        st.subheader("Metric Comparison")
        metrics_to_plot = [c for c in ["PR-AUC", "ROC-AUC", "F1 Score", "Precision", "Recall"] if c in comparison.columns]

        melted = comparison.melt(
            id_vars=["Model"],
            value_vars=metrics_to_plot,
            var_name="Metric",
            value_name="Score",
        )

        fig = px.bar(
            melted, x="Metric", y="Score", color="Model",
            barmode="group",
            title="Model Performance Comparison",
            color_discrete_sequence=["#7F77DD", "#5DCAA5", "#E24B4A"],
            text_auto=".3f",
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, width="stretch")

        # Radar chart
        st.subheader("Radar View")
        fig_radar = go.Figure()
        for _, row in comparison.iterrows():
            values = [row.get(m, 0) for m in metrics_to_plot]
            values.append(values[0])
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics_to_plot + [metrics_to_plot[0]],
                fill="toself",
                name=row["Model"],
                opacity=0.6,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Model Strengths Radar",
        )
        st.plotly_chart(fig_radar, width="stretch")

        # Precision-Recall tradeoff
        if "Precision" in comparison.columns and "Recall" in comparison.columns:
            st.subheader("Precision-Recall Tradeoff")
            fig_pr = px.scatter(
                comparison, x="Recall", y="Precision",
                text="Model", size="F1 Score" if "F1 Score" in comparison.columns else None,
                title="Precision vs Recall by Model",
                color="Model",
                color_discrete_sequence=["#7F77DD", "#5DCAA5", "#E24B4A"],
            )
            fig_pr.update_traces(textposition="top center", marker=dict(size=15))
            fig_pr.update_layout(xaxis_range=[0, 1], yaxis_range=[0, 1])
            st.plotly_chart(fig_pr, width="stretch")

        # Confusion matrix details
        st.subheader("Confusion Matrix Breakdown")
        if all(c in comparison.columns for c in ["True Pos", "False Pos", "False Neg"]):
            for _, row in comparison.iterrows():
                with st.expander(f"📋 {row['Model']}"):
                    cm1, cm2, cm3 = st.columns(3)
                    with cm1:
                        st.metric("True Positives", int(row["True Pos"]))
                    with cm2:
                        st.metric("False Positives", int(row["False Pos"]))
                    with cm3:
                        st.metric("False Negatives", int(row["False Neg"]))

    except ImportError:
        st.error("MLflow not installed. Run: pip install mlflow")
    except Exception as e:
        st.error(f"Could not connect to MLflow: {e}")
        st.info("Make sure MLflow is running at http://localhost:5001")


# ──────────────────────────────────────────────
# Route to selected page
# ──────────────────────────────────────────────
if page == "🔍 Single Prediction":
    page_single_prediction()
elif page == "📁 Batch Prediction":
    page_batch_prediction()
elif page == "📈 Analytics Dashboard":
    page_analytics()
elif page == "🏆 Model Comparison":
    page_model_comparison()