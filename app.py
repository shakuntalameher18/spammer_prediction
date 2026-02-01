import streamlit as st
import pandas as pd
import joblib
import random

# =========================
# Page config
# =========================
st.set_page_config(page_title="Spammer Prediction App", layout="wide")

# =========================
# Custom CSS (BLUE THEME)
# =========================
st.markdown(
    """
    <style>
    div.stFormSubmitButton > button {
        background-color: #1f77b4 !important;   /* blue */
        color: white !important;
        padding: 6px 16px;           /* smaller size */
        font-size: 14px;
        border-radius: 6px;
        border: none;
    }
    div.stFormSubmitButton > button:hover {
        background-color: #135a96 !important;;
        color: white !important;
    }
    /* Accordion header text */
    div[data-testid="stExpander"] summary {
        font-weight: 600;
        background-color: #1f77b4 !important;
        color: white !important;
    }

     div[data-testid="stExpander"] summary:hover  {
        background-color: #135a96 !important;;
        color: white !important;
    }

    

      
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Load model & scaler
# =========================
model = joblib.load("spam_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# Load dataset
# =========================
df = pd.read_csv("fiverr_data.csv")

TARGET_COL = "label"
USER_ID_COL = "user_id"
CONSTANT_COLS = ["X27", "X29", "X30", "X33", "X46", "X47", "X48"]

X_cols = [
    c for c in df.columns
    if c not in [TARGET_COL, USER_ID_COL] + CONSTANT_COLS
]


# =========================
# Choose random slider columns
# =========================
random.seed(42)
NUM_SLIDERS = min(8, len(X_cols))
slider_cols = random.sample(X_cols, NUM_SLIDERS)

# =========================
# Helper: chunk columns
# =========================
def chunk_list(lst, size=10):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

# =========================
# UI
# =========================
with st.container():
    st.markdown(
        """
        <div style="
            background-color:#135a96;
            border-radius:10px;
            text-align:center;
            margin-bottom:10px;
        ">
            <h2 style="color:white; margin:0;">
                📧 Spammer Prediction App
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with st.form("prediction_form"):

    # ---- User ID ----
    st.subheader("👤 User Information")
    user_id = st.text_input("User ID", placeholder="Enter user ID")

    st.markdown("---")
    st.subheader("🔧 Feature Inputs")

    input_data = {}

    for idx, group in enumerate(chunk_list(X_cols, 10), start=1):
        with st.expander(f"Feature Group {idx}", expanded=(idx == 1)):

            cols = st.columns(2)

            for i, col in enumerate(group):
                with cols[i % 2]:

                    series = df[col].dropna()

                    if series.empty:
                        input_data[col] = 0
                        continue

                    col_min = series.min()
                    col_max = series.max()
                    col_default = series.median()

                    use_slider = col in slider_cols and col_min != col_max

                    # Float column
                    if col == "X13":
                        if use_slider:
                            input_data[col] = st.slider(
                                col,
                                float(col_min),
                                float(col_max),
                                float(col_default),
                                step=0.01
                            )
                        else:
                            input_data[col] = st.number_input(
                                col,
                                value=float(col_default),
                                format="%.4f"
                            )
                    # Integer columns
                    else:
                        if use_slider:
                            input_data[col] = st.slider(
                                col,
                                int(col_min),
                                int(col_max),
                                int(col_default),
                                step=1
                            )
                        else:
                            input_data[col] = st.number_input(
                                col,
                                value=int(col_default),
                                step=1
                            )

    # ---- Predict button (NO reload until click) ----
    submitted = st.form_submit_button("🚀 Predict")

# =========================
# Prediction
# =========================
if submitted:

    if user_id.strip() == "":
        st.warning("⚠️ Please enter a User ID.")
    else:
        input_df = pd.DataFrame([input_data])[X_cols]
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)

        st.markdown("---")
        st.subheader("📊 Prediction Result")
        st.write(f"**User ID:** `{user_id}`")

        if prediction == 1:
            st.error("🚨 Prediction: **SPAM**")
        else:
            st.success("✅ Prediction: **NOT SPAM**")

        st.write("### 🔢 Prediction Probabilities")
        st.write(prediction_proba)
