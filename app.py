import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# 2. CUSTOM STYLING
# =====================================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1250px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .main-title {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0;
        }
        .subtitle {
            color: #9CA3AF;
            font-size: 1.05rem;
            margin-top: 0.4rem;
            margin-bottom: 1.5rem;
        }
        .section-title {
            font-size: 1.4rem;
            font-weight: 700;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .risk-card {
            padding: 1.2rem;
            border-radius: 12px;
            margin: 1rem 0;
        }
        .risk-low {
            background-color: rgba(34, 197, 94, 0.12);
            border: 1px solid #22C55E;
        }
        .risk-medium {
            background-color: rgba(245, 158, 11, 0.12);
            border: 1px solid #F59E0B;
        }
        .risk-high {
            background-color: rgba(239, 68, 68, 0.12);
            border: 1px solid #EF4444;
        }
        div[data-testid="stMetric"] {
            background-color: rgba(128, 128, 128, 0.07);
            border: 1px solid rgba(128, 128, 128, 0.2);
            padding: 1rem;
            border-radius: 12px;
        }
        div.stButton > button {
            width: 100%;
            min-height: 3rem;
            font-weight: 700;
            border-radius: 10px;
        }
        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# 3. LOAD TRAINED MODEL
# =====================================================

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "churnguard_model.pkl"
DATA_PATH = BASE_DIR / "telco_customer_churn_clean.csv"
TEST_DATA_PATH = BASE_DIR / "churnguard_test_data.csv"


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(
        "Model file not found. Make sure "
        "'churnguard_model.pkl' is in the same folder as app.py."
    )
    st.stop()
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.info(
        "Check that the installed scikit-learn version is compatible "
        "with the version used to save the model in Google Colab."
    )
    st.stop()

# =====================================================
# 4. LOAD OPTIONAL BUSINESS ANALYTICS DATA
# =====================================================


@st.cache_data
def load_customer_data(file_path):
    df = pd.read_csv(file_path)

    for column in ["Churn Value", "Tenure Months", "Monthly Charges", "Total Charges"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


df_analytics = None

if DATA_PATH.exists():
    try:
        df_analytics = load_customer_data(DATA_PATH)
    except Exception as error:
        st.warning(f"Could not load the analytics dataset: {error}")
else:
    st.info(
        "The original Excel dataset is not included in this deployment. "
        "Individual prediction, model performance, and feature importance "
        "remain available; dataset-based analytics are disabled."
    )

# =====================================================
# 5. HEADER
# =====================================================

st.markdown(
    '<div class="main-title">📊 ChurnGuard AI</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Customer Churn Prediction & Retention Analytics</div>',
    unsafe_allow_html=True,
)
st.write(
    "Estimate customer churn risk, understand potential risk factors, "
    "and explore retention opportunities using machine learning."
)
st.divider()

# =====================================================
# 6. SIDEBAR
# =====================================================

with st.sidebar:
    st.title("🛡️ ChurnGuard AI")
    st.caption("Customer Intelligence Platform")
    st.divider()
    st.subheader("About this project")
    st.write(
        "ChurnGuard AI uses a trained Random Forest classification "
        "pipeline to estimate the likelihood that a customer will leave."
    )
    st.markdown("**Model:** Random Forest")
    st.markdown("**Dataset:** IBM Telco Customer Churn")
    st.markdown("**Input features:** 19")
    st.markdown("**Prediction:** Churn / No Churn")
    st.divider()
    st.caption(
        "This application is a portfolio project. Predictions are "
        "estimates and should not be treated as guarantees."
    )

# =====================================================
# 7. CUSTOMER INPUTS
# =====================================================

st.markdown(
    '<div class="section-title">👤 Customer Information</div>',
    unsafe_allow_html=True,
)

with st.form("customer_prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Personal Details")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda value: "Yes" if value == 1 else "No",
        )
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input(
            "Tenure (Months)", min_value=0, max_value=72, value=2, step=1
        )

    with col2:
        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )
        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )

    with col3:
        st.subheader("Contract & Billing")
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    st.divider()
    st.subheader("💳 Billing Information")
    bill_col1, bill_col2 = st.columns(2)

    with bill_col1:
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=1000.0,
            value=85.0,
            step=1.0,
        )

    with bill_col2:
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=100000.0,
            value=170.0,
            step=10.0,
        )

    submitted = st.form_submit_button(
        "🔍 Predict Customer Churn",
        type="primary",
        use_container_width=True,
    )

# =====================================================
# 8. MAKE PREDICTION
# =====================================================

if submitted:
    customer = pd.DataFrame(
        [
            {
                "Gender": gender,
                "Senior Citizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "Tenure Months": tenure,
                "Phone Service": phone_service,
                "Multiple Lines": multiple_lines,
                "Internet Service": internet_service,
                "Online Security": online_security,
                "Online Backup": online_backup,
                "Device Protection": device_protection,
                "Tech Support": tech_support,
                "Streaming TV": streaming_tv,
                "Streaming Movies": streaming_movies,
                "Contract": contract,
                "Paperless Billing": paperless_billing,
                "Payment Method": payment_method,
                "Monthly Charges": monthly_charges,
                "Total Charges": total_charges,
            }
        ]
    )

    try:
        probabilities = model.predict_proba(customer)[0]
        classes = list(model.classes_)
        churn_index = classes.index(1)
        churn_probability = float(probabilities[churn_index])
        retention_probability = 1.0 - churn_probability
        prediction = int(model.predict(customer)[0])

        if churn_probability < 0.30:
            risk_level, risk_class, risk_emoji = "Low Risk", "risk-low", "🟢"
        elif churn_probability < 0.60:
            risk_level, risk_class, risk_emoji = (
                "Medium Risk", "risk-medium", "🟡"
            )
        else:
            risk_level, risk_class, risk_emoji = "High Risk", "risk-high", "🔴"

        st.divider()
        st.markdown(
            '<div class="section-title">📈 Prediction Results</div>',
            unsafe_allow_html=True,
        )
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Churn Probability", f"{churn_probability:.1%}")
        metric2.metric("Retention Probability", f"{retention_probability:.1%}")
        metric3.metric("Customer Risk Level", risk_level)

        st.markdown(
            f"""
            <div class="risk-card {risk_class}">
                <h3>{risk_emoji} {risk_level}</h3>
                <p>Estimated churn probability:
                <strong>{churn_probability:.1%}</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(churn_probability)

        if prediction == 1:
            st.error("⚠️ The model classifies this customer as likely to churn.")
        else:
            st.success("✅ The model classifies this customer as likely to stay.")

        st.markdown(
            '<div class="section-title">💡 Suggested Retention Actions</div>',
            unsafe_allow_html=True,
        )
        recommendations = []

        if contract == "Month-to-month":
            recommendations.append(
                "Consider offering a suitable long-term contract with clear pricing and benefits."
            )
        if tenure <= 12:
            recommendations.append(
                "Provide onboarding support and consider an appropriate early-tenure loyalty incentive."
            )
        if online_security == "No" and internet_service != "No":
            recommendations.append(
                "Explain available online security options and offer a suitable plan if relevant."
            )
        if tech_support == "No" and internet_service != "No":
            recommendations.append(
                "Offer technical support options to help resolve potential service concerns."
            )
        if payment_method == "Electronic check":
            recommendations.append(
                "Offer convenient automatic payment options, without pressuring the customer to change."
            )
        if monthly_charges >= 70:
            recommendations.append(
                "Review the customer's plan and explain any suitable lower-cost alternatives or available discounts."
            )
        if not recommendations:
            recommendations.append(
                "Maintain service quality and contact the customer periodically to understand their needs."
            )

        for recommendation in recommendations:
            st.markdown(f"- {recommendation}")

        with st.expander("View customer input summary"):
            summary = pd.DataFrame(
                {"Feature": customer.columns, "Value": customer.iloc[0].values}
            )
            st.dataframe(summary, use_container_width=True, hide_index=True)

        st.caption(
            "Risk bands are configurable business thresholds. Recommendations are "
            "rule-based suggestions, not explanations of the model's internal reasoning. "
            "Model probabilities may not be perfectly calibrated."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.info(
            "Check that the saved model was trained using the same 19 input features "
            "and that feature names and data types match the training pipeline."
        )

# =====================================================
# 9. BUSINESS ANALYTICS + CUSTOMER RISK SEGMENTATION
#    These sections require the optional Excel dataset.
# =====================================================

st.divider()
st.header("📊 Business Analytics Dashboard")
st.write(
    "Explore overall customer churn, contract patterns, customer tenure, and billing trends."
)

if df_analytics is None:
    st.warning(
        "Business Analytics and Customer Risk Segmentation are unavailable because "
        "'Telco_customer_churn.xlsx' is not present in this deployment."
    )
    st.info(
        "To enable these sections online, provide an appropriate dataset. "
        "Keep customer identifiers and precise location details out of any public dataset."
    )
else:
    analytics_df = df_analytics.copy()
    analytics_df = analytics_df.dropna(
        subset=["Churn Value", "Tenure Months", "Monthly Charges"]
    )

    analytics_df["Churn Status"] = analytics_df["Churn Value"].map(
        {0: "Retained", 1: "Churned"}
    )
    analytics_df["Tenure Group"] = pd.cut(
        analytics_df["Tenure Months"],
        bins=[-1, 12, 24, 48, 60, 72],
        labels=[
            "0–12 months",
            "13–24 months",
            "25–48 months",
            "49–60 months",
            "61–72 months",
        ],
    )

    total_customers = len(analytics_df)
    churned_customers = int((analytics_df["Churn Value"] == 1).sum())
    retained_customers = int((analytics_df["Churn Value"] == 0).sum())
    churn_rate = (
        churned_customers / total_customers * 100 if total_customers else 0
    )
    average_monthly_charges = analytics_df["Monthly Charges"].mean()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Customers", f"{total_customers:,}")
    kpi2.metric("Churned Customers", f"{churned_customers:,}")
    kpi3.metric("Retained Customers", f"{retained_customers:,}")
    kpi4.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
    st.caption(
        f"Average monthly charge across customers: ${average_monthly_charges:.2f}"
    )
    st.divider()

    st.subheader("🔎 Explore Customer Segments")
    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        contract_options = sorted(
            analytics_df["Contract"].dropna().unique().tolist()
        )
        selected_contracts = st.multiselect(
            "Filter by contract",
            options=contract_options,
            default=contract_options,
        )

    with filter_col2:
        internet_options = sorted(
            analytics_df["Internet Service"].dropna().unique().tolist()
        )
        selected_internet = st.multiselect(
            "Filter by internet service",
            options=internet_options,
            default=internet_options,
        )

    filtered_df = analytics_df[
        analytics_df["Contract"].isin(selected_contracts)
        & analytics_df["Internet Service"].isin(selected_internet)
    ]

    if filtered_df.empty:
        st.warning("No customers match the selected filters.")
    else:
        filtered_churn_rate = filtered_df["Churn Value"].mean() * 100
        st.metric("Churn Rate for Selected Segment", f"{filtered_churn_rate:.1f}%")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.subheader("Customer Churn Distribution")
            churn_counts = (
                filtered_df["Churn Status"]
                .value_counts()
                .rename_axis("Customer Status")
                .reset_index(name="Customers")
            )
            fig_churn = px.pie(
                churn_counts,
                names="Customer Status",
                values="Customers",
                hole=0.55,
                title="Retained vs Churned Customers",
            )
            st.plotly_chart(fig_churn, use_container_width=True)

        with chart_col2:
            st.subheader("Churn Rate by Contract")
            contract_churn = (
                filtered_df.groupby("Contract", observed=True)["Churn Value"]
                .mean()
                .mul(100)
                .reset_index(name="Churn Rate (%)")
            )
            fig_contract = px.bar(
                contract_churn,
                x="Contract",
                y="Churn Rate (%)",
                text_auto=".1f",
                title="Percentage of Customers Who Churned",
            )
            fig_contract.update_layout(
                yaxis_title="Churn Rate (%)", xaxis_title="Contract Type"
            )
            st.plotly_chart(fig_contract, use_container_width=True)

        st.subheader("Churn Rate by Customer Tenure")
        tenure_churn = (
            filtered_df.groupby("Tenure Group", observed=True)["Churn Value"]
            .mean()
            .mul(100)
            .reset_index(name="Churn Rate (%)")
        )
        fig_tenure = px.bar(
            tenure_churn,
            x="Tenure Group",
            y="Churn Rate (%)",
            text_auto=".1f",
            title="Churn Rate Across Tenure Groups",
        )
        fig_tenure.update_layout(
            xaxis_title="Customer Tenure", yaxis_title="Churn Rate (%)"
        )
        st.plotly_chart(fig_tenure, use_container_width=True)

        st.subheader("Monthly Charges and Churn")
        fig_charges = px.box(
            filtered_df,
            x="Churn Status",
            y="Monthly Charges",
            color="Churn Status",
            points=False,
            title="Monthly Charges: Retained vs Churned Customers",
        )
        st.plotly_chart(fig_charges, use_container_width=True)

    st.divider()
    st.subheader("💡 Why Are Customers Leaving?")
    churned_df = analytics_df[analytics_df["Churn Value"] == 1].copy()
    reason_df = churned_df.dropna(subset=["Churn Reason"])

    if not reason_df.empty:
        reason_counts = reason_df["Churn Reason"].value_counts().reset_index()
        reason_counts.columns = ["Churn Reason", "Customers"]
        fig_reasons = px.bar(
            reason_counts.head(10),
            x="Customers",
            y="Churn Reason",
            orientation="h",
            text="Customers",
            title="Top Customer Churn Reasons",
        )
        fig_reasons.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_reasons, use_container_width=True)
        st.caption(
            "This chart includes only churned customers with a recorded churn reason. "
            "It does not establish the cause of every churn event."
        )
    else:
        st.info("No recorded churn reasons are available.")

    # =====================================================
    # CUSTOMER RISK SEGMENTATION
    # =====================================================

    st.divider()
    st.header("🎯 Customer Risk Segmentation")
    st.write(
        "Use the trained machine learning model to estimate churn risk for customers "
        "in the loaded dataset."
    )

    feature_columns = [
        "Gender",
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Tenure Months",
        "Phone Service",
        "Multiple Lines",
        "Internet Service",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Contract",
        "Paperless Billing",
        "Payment Method",
        "Monthly Charges",
        "Total Charges",
    ]

    if st.button("Analyze Customer Risk", type="primary"):
        missing_columns = [
            col for col in feature_columns if col not in df_analytics.columns
        ]

        if missing_columns:
            st.error(f"Missing required model features: {missing_columns}")
        else:
            with st.spinner("Analyzing customer records..."):
                batch_df = df_analytics[feature_columns].copy()
                batch_df["Total Charges"] = pd.to_numeric(
                    batch_df["Total Charges"], errors="coerce"
                )
                # The pipeline's imputer can handle missing numeric values;
                # don't replace missing charges with zero unless that matches training.
                probabilities = model.predict_proba(batch_df)
                classes = list(model.classes_)
                churn_index = classes.index(1)
                churn_probabilities = probabilities[:, churn_index]
                predictions = model.predict(batch_df)

                results = df_analytics.copy()
                results["Predicted Churn Probability"] = churn_probabilities
                results["Predicted Churn"] = predictions
                results["Risk Level"] = pd.cut(
                    results["Predicted Churn Probability"],
                    bins=[-0.001, 0.30, 0.60, 1.001],
                    labels=["Low", "Medium", "High"],
                )
                st.session_state["churn_risk_results"] = results

    if "churn_risk_results" in st.session_state:
        results = st.session_state["churn_risk_results"]
        high_risk = int((results["Risk Level"] == "High").sum())
        medium_risk = int((results["Risk Level"] == "Medium").sum())
        low_risk = int((results["Risk Level"] == "Low").sum())

        st.subheader("Customer Risk Overview")
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        risk_col1.metric("🔴 High Risk", f"{high_risk:,}")
        risk_col2.metric("🟡 Medium Risk", f"{medium_risk:,}")
        risk_col3.metric("🟢 Low Risk", f"{low_risk:,}")
        st.caption(
            "Risk bands are configurable business thresholds, not categories learned by the model."
        )

        st.subheader("Customer Risk Report")
        risk_filter = st.multiselect(
            "Filter by risk level",
            ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
        )
        report_df = results[
            results["Risk Level"].astype(str).isin(risk_filter)
        ].copy()

        display_columns = [
        col for col in [
        "CustomerID",
        "Contract",
        "Tenure Months",
        "Monthly Charges",
        "Predicted Churn Probability",
        "Predicted Churn",
        "Risk Level",
        ]
        if col in results.columns
        ]
        available_display_columns = [
            col for col in display_columns if col in report_df.columns
        ]
        report_df["Predicted Churn Probability"] = (
            report_df["Predicted Churn Probability"] * 100
        ).round(1)

        st.dataframe(
            report_df[available_display_columns].sort_values(
                "Predicted Churn Probability", ascending=False
            ),
            use_container_width=True,
            hide_index=True,
        )
        csv_data = report_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Customer Risk Report",
            data=csv_data,
            file_name="churnguard_customer_risk_report.csv",
            mime="text/csv",
        )

# =====================================================
# 10. MODEL PERFORMANCE DASHBOARD
# =====================================================

st.divider()
st.header("🧪 Model Performance Dashboard")
st.write(
    "Evaluate the Random Forest model using the held-out test data "
    "from the original training experiment."
)

if TEST_DATA_PATH.exists():
    try:
        test_df = pd.read_csv(TEST_DATA_PATH)

        if "Actual Churn" not in test_df.columns:
            st.error("The test CSV must contain an 'Actual Churn' column.")
        else:
            y_test_eval = test_df["Actual Churn"].astype(int)
            X_test_eval = test_df.drop(columns=["Actual Churn"])

            expected_features = (
                list(model.feature_names_in_)
                if hasattr(model, "feature_names_in_")
                else None
            )

            if expected_features is not None:
                missing = [
                    col for col in expected_features if col not in X_test_eval.columns
                ]
                if missing:
                    st.error(f"Missing test features: {missing}")
                    st.stop()
                X_test_eval = X_test_eval[expected_features]

            y_pred_eval = model.predict(X_test_eval)
            y_prob_eval = model.predict_proba(X_test_eval)
            class_labels = list(model.classes_)
            churn_index = class_labels.index(1)
            churn_scores = y_prob_eval[:, churn_index]

            accuracy = accuracy_score(y_test_eval, y_pred_eval)
            precision = precision_score(
                y_test_eval, y_pred_eval, pos_label=1, zero_division=0
            )
            recall = recall_score(
                y_test_eval, y_pred_eval, pos_label=1, zero_division=0
            )
            f1 = f1_score(
                y_test_eval, y_pred_eval, pos_label=1, zero_division=0
            )
            auc = roc_auc_score(y_test_eval, churn_scores)

            metric1, metric2, metric3 = st.columns(3)
            metric4, metric5 = st.columns(2)
            metric1.metric("Accuracy", f"{accuracy:.2%}")
            metric2.metric("Precision", f"{precision:.2%}")
            metric3.metric("Recall", f"{recall:.2%}")
            metric4.metric("F1 Score", f"{f1:.2%}")
            metric5.metric("ROC-AUC", f"{auc:.3f}")

            st.caption(
                "Precision, recall, and F1 treat churn (class 1) as the positive class."
            )
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test_eval, y_pred_eval, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
            display = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["Retained", "Churned"],
            )
            display.plot(ax=ax, values_format="d", colorbar=False)
            ax.set_title("Actual vs Predicted Customer Churn", fontsize=10)
            ax.set_xlabel("Predicted label", fontsize=9)
            ax.set_ylabel("True label", fontsize=9)
            ax.tick_params(axis="both", labelsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

            st.subheader("Classification Report")
            report = classification_report(
                y_test_eval,
                y_pred_eval,
                labels=[0, 1],
                target_names=["Retained", "Churned"],
                output_dict=True,
                zero_division=0,
            )
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(3), use_container_width=True)

    except Exception as error:
        st.error(f"Could not calculate model metrics: {error}")
else:
    st.info(
        "Model performance metrics are unavailable because "
        "'churnguard_test_data.csv' is not present in this deployment."
    )

# =====================================================
# 11. FEATURE IMPORTANCE DASHBOARD
# =====================================================

st.divider()
st.header("🔍 What Drives Customer Churn?")
st.write(
    "Explore which input features the trained Random Forest uses most when making predictions."
)

try:
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importance_values = classifier.feature_importances_

    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importance_values}
    )
    importance_df["Feature"] = (
        importance_df["Feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
        .str.replace("_", " ", regex=False)
    )
    importance_df = importance_df.sort_values("Importance", ascending=False)

    top_n = st.slider(
        "Number of features to display",
        min_value=5,
        max_value=max(5, min(19, len(importance_df))),
        value=max(5, min(10, len(importance_df))),
    )
    top_features = importance_df.head(top_n).sort_values(
        "Importance", ascending=True
    )
    fig_importance = px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"Top {top_n} Features by Importance",
        text=top_features["Importance"].map(lambda value: f"{value:.3f}"),
    )
    fig_importance.update_layout(
        xaxis_title="Importance Score",
        yaxis_title="Customer Feature",
        height=max(350, top_n * 35),
    )
    st.plotly_chart(fig_importance, use_container_width=True)
    st.info(
        "Feature importance indicates how much a feature contributes to the Random Forest's "
        "overall decision process. It does not show causation or explain an individual prediction."
    )
except Exception as error:
    st.error(f"Could not generate feature importance: {error}")

# =====================================================
# 12. FOOTER
# =====================================================

st.divider()
st.caption("ChurnGuard AI | Customer Churn Prediction & Retention Analytics")
