import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

THRESHOLD = 0.40

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1150px;}
    .header {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        padding: 1.6rem 2rem; border-radius: 14px; color: white; margin-bottom: 1.5rem;
    }
    .header h1 {margin: 0; font-size: 2rem;}
    .header p {margin: .3rem 0 0; opacity: .9;}
    div.stButton > button {
        width: 100%; border: none; border-radius: 10px; padding: .7rem;
        font-weight: 600; color: white; background: #4f46e5;
    }
    div.stButton > button:hover {background: #4338ca; color: white;}
    footer, #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load("customer_churn_xgb_pipeline.pkl")


try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file 'customer_churn_xgb_pipeline.pkl' not found.")
    st.stop()


def get_risk(prob):
    if prob >= 0.65:
        return "High risk", "#ef4444"
    if prob >= THRESHOLD:
        return "Medium risk", "#f59e0b"
    return "Low risk", "#22c55e"


def make_gauge(prob, color):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, THRESHOLD * 100], "color": "rgba(34,197,94,.2)"},
                    {"range": [THRESHOLD * 100, 65], "color": "rgba(245,158,11,.2)"},
                    {"range": [65, 100], "color": "rgba(239,68,68,.2)"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def get_flags(data):
    flags = []
    if data["IsActiveMember"] == 0:
        flags.append("Customer is inactive")
    if data["Age"] >= 45:
        flags.append("Age is 45 or above")
    if data["NumOfProducts"] == 1:
        flags.append("Uses only 1 product")
    if data["NumOfProducts"] >= 3:
        flags.append("Holds 3 or more products")
    if data["Geography"] == "Germany":
        flags.append("Based in Germany")
    if data["Balance"] > 100000 and data["IsActiveMember"] == 0:
        flags.append("High balance but inactive account")
    if data["Satisfaction Score"] <= 2:
        flags.append("Low satisfaction score")
    return flags


with st.sidebar:
    st.header("About")
    st.write("Predicts the probability of customer churn using an XGBoost pipeline.")
    st.write(f"A customer is flagged as high risk when the churn probability is {int(THRESHOLD * 100)}% or more.")
    st.caption("Built with Python, scikit-learn, XGBoost and Streamlit")

st.markdown(
    """
    <div class="header">
        <h1>Customer Churn Predictor</h1>
        <p>Enter the customer's details to see the churn probability.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.1, 1], gap="large")

with left:
    tab1, tab2, tab3 = st.tabs(["Profile", "Banking", "Engagement"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
            age = st.slider("Age", 18, 100, 35)
        with c2:
            gender = st.selectbox("Gender", ["Female", "Male"])
            tenure = st.slider("Tenure (Years)", 0, 10, 5)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            credit_score = st.number_input("Credit Score", 300, 850, 650)
            balance = st.number_input("Account Balance", min_value=0.0, value=50000.0, step=1000.0)
            card_type = st.selectbox("Card Type", ["DIAMOND", "GOLD", "SILVER", "PLATINUM"])
        with c2:
            estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.0, step=1000.0)
            num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
            has_cr_card = st.checkbox("Has Credit Card", value=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            is_active_member = st.checkbox("Active Member", value=True)
            satisfaction_score = st.select_slider("Satisfaction Score", options=[1, 2, 3, 4, 5], value=3)
        with c2:
            point_earned = st.number_input("Points Earned", min_value=0, value=500, step=10)

    predict = st.button("Predict Churn Risk")

customer = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "HasCrCard": int(has_cr_card),
    "IsActiveMember": int(is_active_member),
    "EstimatedSalary": estimated_salary,
    "Satisfaction Score": satisfaction_score,
    "Card Type": card_type,
    "Point Earned": point_earned,
}

with right:
    if predict:
        prob = float(model.predict_proba(pd.DataFrame([customer]))[0][1])
        label, color = get_risk(prob)

        st.subheader("Prediction Result")
        st.plotly_chart(make_gauge(prob, color), use_container_width=True)

        col1, col2 = st.columns(2)
        col1.metric("Churn Probability", f"{prob * 100:.2f}%")
        col2.metric("Risk Level", label)

        if prob >= THRESHOLD:
            st.error("Action is recommended to retain this customer.")
        else:
            st.success("This customer looks stable.")

        flags = get_flags(customer)
        if flags:
            st.write("**Things to note:**")
            for f in flags:
                st.write(f"- {f}")
    else:
        st.info("Fill in the details on the left and click Predict Churn Risk.")