import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------
# CACHED LOADERS
# --------------------------

@st.cache_resource
def load_model():
    model = joblib.load("fraud_model.pkl")
    return model

@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv")
    return df

# --------------------------
# LOAD MODEL & DATA
# --------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection App")
st.write(
    "This app uses a trained **Random Forest** model to detect whether a transaction "
    "is **Fraud (1)** or **Safe (0)** based on the famous Kaggle credit card dataset."
)

model = load_model()
data = load_data()

X = data.drop("Class", axis=1)
y = data["Class"]

# --------------------------
# SIDEBAR
# --------------------------

st.sidebar.header("Options")

mode = st.sidebar.radio(
    "Choose Mode",
    ["Random Transaction (Recommended)", "Manual Input (Advanced)"]
)

sample_type = st.sidebar.selectbox(
    "Sample Type (only for Random mode)",
    ["Any", "Only Safe (Class = 0)", "Only Fraud (Class = 1)"],
    index=0
)

show_raw = st.sidebar.checkbox("Show raw transaction row", value=True)

st.sidebar.markdown("---")
st.sidebar.write("Model: **RandomForestClassifier**")
st.sidebar.write("Trained with undersampling + scaling + SHAP analysis.")

# --------------------------
# HELPER: PREDICT FUNCTION
# --------------------------

def predict_and_show(input_df):
    """Takes a dataframe with same columns as training X and shows prediction nicely."""
    proba = model.predict_proba(input_df)[0]
    pred = int(model.predict(input_df)[0])

    fraud_prob = proba[1]   # probability for class 1 (fraud)
    safe_prob = proba[0]    # probability for class 0 (safe)

    st.subheader("🔮 Model Prediction")

    col1, col2 = st.columns(2)

    with col1:
        if pred == 1:
            st.error(f"⚠️ PREDICTION: FRAUD (Class = 1)")
        else:
            st.success(f"✅ PREDICTION: SAFE (Class = 0)")

    with col2:
        st.metric("Fraud Probability (Class 1)", f"{fraud_prob*100:.2f}%")
        st.metric("Safe Probability (Class 0)", f"{safe_prob*100:.2f}%")

    st.markdown("---")


# --------------------------
# MODE 1: RANDOM TRANSACTION
# --------------------------

if mode == "Random Transaction (Recommended)":
    st.subheader("🎲 Random Transaction from Dataset")

    # Filter based on sidebar selection
    if sample_type == "Any":
        df_filtered = data.copy()
    elif sample_type == "Only Safe (Class = 0)":
        df_filtered = data[data["Class"] == 0]
    else:
        df_filtered = data[data["Class"] == 1]

    if df_filtered.empty:
        st.warning("No samples available for the selected type.")
    else:
        if st.button("🔁 Pick a Random Transaction"):
            sample = df_filtered.sample(1, random_state=None)
            sample_X = sample.drop("Class", axis=1)

            st.write("**Selected Transaction Details:**")
            if show_raw:
                st.dataframe(sample)

            predict_and_show(sample_X)

            # Show info about actual label
            actual_class = int(sample["Class"].values[0])
            if actual_class == 1:
                st.info("ℹ️ NOTE: This transaction is actually **FRAUD (Class = 1)** in the dataset.")
            else:
                st.info("ℹ️ NOTE: This transaction is actually **SAFE (Class = 0)** in the dataset.")

        else:
            st.info("Click **'Pick a Random Transaction'** to see a prediction.")


# --------------------------
# MODE 2: MANUAL INPUT
# --------------------------

elif mode == "Manual Input (Advanced)":
    st.subheader("🧮 Manual Feature Input (Advanced Users)")

    st.write(
        "You can manually provide feature values. "
        "Realistic values come from the `creditcard.csv` dataset (scaled Time & Amount, PCA V1-V28). "
        "This mode is mainly for experimentation."
    )

    with st.expander("ℹ️ Tip: Leave default values or tweak slightly for testing.", expanded=False):
        st.write("For real-world scenarios, features V1–V28 are PCA components, not directly interpretable.")

    # Layout for inputs
    col_left, col_right = st.columns(2)

    with col_left:
        time_val = st.number_input("Time (scaled)", value=0.0)
        amount_val = st.number_input("Amount (scaled)", value=0.0)

    v_values = []

    with col_left:
        for i in range(1, 15):
            v = st.number_input(f"V{i}", value=0.0, key=f"V{i}_L")
            v_values.append(v)

    with col_right:
        for i in range(15, 29):
            v = st.number_input(f"V{i}", value=0.0, key=f"V{i}_R")
            v_values.append(v)

    # Make dataframe for model
    input_features = [time_val, amount_val] + v_values
    columns = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
    input_df = pd.DataFrame([input_features], columns=columns)

    if st.button("🚀 Predict from Manual Input"):
        st.write("**Manual Input Features:**")
        st.dataframe(input_df)

        predict_and_show(input_df)

# --------------------------
# FOOTER
# --------------------------

st.markdown("---")
st.caption("Built by omShukla69 · Credit Card Fraud Detection · RandomForest + Explainable AI (SHAP)")
