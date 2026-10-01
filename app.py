import streamlit as st
import pandas as pd
import base64

from helper import opener


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ScaniaGuard",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# BACKGROUND IMAGE
# =========================================================

def set_background(image_path):

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(
            image_file.read()
        ).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{
            background-image:
                linear-gradient(
                    rgba(5, 10, 20, 0.88),
                    rgba(5, 10, 20, 0.94)
                ),
                url("data:image/jpeg;base64,{encoded_image}");

            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


set_background("assets/truck_image_1.jpg")


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* =========================================
       MAIN PAGE
       ========================================= */

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* =========================================
       SIDEBAR
       ========================================= */

    section[data-testid="stSidebar"] {
        background-color: rgba(7, 13, 25, 0.97);
    }


    /* =========================================
       HEADINGS
       ========================================= */

    h1 {
        font-weight: 800 !important;
        letter-spacing: 2px;
    }

    h2 {
        font-weight: 750 !important;
    }

    h3 {
        font-weight: 700 !important;
    }


    /* =========================================
       NATIVE STREAMLIT CONTAINERS
       ========================================= */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(15, 23, 42, 0.72);
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.10);
        backdrop-filter: blur(10px);
    }


    /* =========================================
       METRIC CARDS
       ========================================= */

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 16px;
        padding: 18px;
    }

    div[data-testid="stMetricValue"] {
        font-weight: 800;
    }


    /* =========================================
       BUTTON
       ========================================= */

    .stButton > button {
        width: 100%;
        height: 3.2rem;
        border-radius: 12px;
        font-size: 1rem;
        font-weight: 700;
    }


    /* =========================================
       FILE UPLOADER
       ========================================= */

    div[data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.65);
        border-radius: 15px;
        padding: 10px;
    }


    /* =========================================
       RADIO BUTTON AREA
       ========================================= */

    div[role="radiogroup"] {
        background: rgba(15, 23, 42, 0.65);
        padding: 12px 18px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.10);
    }


    /* =========================================
       DATAFRAME
       ========================================= */

    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }


    /* =========================================
       DOWNLOAD BUTTON
       ========================================= */

    .stDownloadButton > button {
        width: 100%;
        height: 3rem;
        border-radius: 12px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HERO SECTION
# =========================================================

st.title("🚛 SCANIAGUARD")

st.markdown(
    "### Cost-Sensitive APS Failure Detection"
)

st.caption(
    "Machine Learning System for Predictive Truck Failure Detection"
)


# =========================================================
# MODEL STATUS
# =========================================================

# st.markdown("### Model Status")

# status1, status2, status3, status4 = st.columns(4)


# with status1:

#     st.metric(
#         label="🟢 Model Status",
#         value="ONLINE"
#     )


# with status2:

#     st.metric(
#         label="🤖 ML Model",
#         value="XGBoost"
#     )


# with status3:

#     st.metric(
#         label="📊 Features",
#         value="236"
#     )


# with status4:

#     st.metric(
#         label="🎯 Threshold",
#         value="0.024"
#     )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🚛 ScaniaGuard")

    st.divider()
    st.subheader('Evaluation Metric')
    st.write("**False Positive Cost:** ₹10")
    st.write("**False Negative Cost:** ₹500")

    st.divider()

    st.subheader("How It Works")

    st.write(
        """
        **1.** Upload truck data

        **2.** Handle missing values

        **3.** Generate missingness indicators

        **4.** Run XGBoost

        **5.** Apply cost-sensitive threshold

        **6.** Display prediction
        """
    )

    st.divider()

    st.success("Model Ready")


# =========================================================
# INTRODUCTION
# =========================================================

with st.container(border=True):

    st.subheader("🔍 APS Failure Detection")

    st.write(
        """
        Upload Scania truck operational data and use the trained
        machine-learning model to identify trucks that may belong
        to the APS-failure class.
        """
    )


# =========================================================
# MODE SELECTION
# =========================================================

st.markdown("### Select Analysis Mode")

mode = st.radio(
    "Choose how you want to use ScaniaGuard:",
    [
        "🚛 Predict New Trucks",
        "📊 Evaluate Labeled Data"
    ],
    horizontal=True
)


# =========================================================
# FILE UPLOAD
# =========================================================

st.markdown("### Upload Truck Data")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"],
    help=(
        "Upload a Scania CSV containing the required "
        "operational features."
    )
)


# =========================================================
# ANALYSIS
# =========================================================

if uploaded_file is not None:

    st.success(
        f"✅ File uploaded: {uploaded_file.name}"
    )

    if st.button("🚀 Run ScaniaGuard Analysis"):

        try:

            # =================================================
            # MODEL PREDICTION
            # =================================================

            predictions, probabilities, y_true = opener(
                uploaded_file
            )

            total_trucks = len(predictions)

            predicted_failures = int(
                (predictions == 1).sum()
            )

            predicted_normal = int(
                (predictions == 0).sum()
            )


            # =================================================
            # SUMMARY
            # =================================================

            st.markdown("---")

            st.header("📊 Analysis Summary")

            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Total Trucks",
                    f"{total_trucks:,}"
                )


            with col2:

                st.metric(
                    "🚨 APS Failures Detected",
                    f"{predicted_failures:,}"
                )


            with col3:

                st.metric(
                    "✅ Not APS Failure",
                    f"{predicted_normal:,}"
                )


            # =================================================
            # EVALUATION MODE
            # =================================================

            if mode == "📊 Evaluate Labeled Data":

                st.header("📈 Model Evaluation")

                if y_true is None:

                    st.warning(
                        """
                        This CSV does not contain a `class`
                        column. Evaluation requires actual
                        labels.
                        """
                    )

                else:

                    # -----------------------------------------
                    # CONFUSION MATRIX VALUES
                    # -----------------------------------------

                    tp = int(
                        (
                            (y_true == 1) &
                            (predictions == 1)
                        ).sum()
                    )

                    tn = int(
                        (
                            (y_true == 0) &
                            (predictions == 0)
                        ).sum()
                    )

                    fp = int(
                        (
                            (y_true == 0) &
                            (predictions == 1)
                        ).sum()
                    )

                    fn = int(
                        (
                            (y_true == 1) &
                            (predictions == 0)
                        ).sum()
                    )


                    # -----------------------------------------
                    # COST
                    # -----------------------------------------

                    fp_cost = fp * 10
                    fn_cost = fn * 500

                    total_cost = (
                        fp_cost +
                        fn_cost
                    )


                    # -----------------------------------------
                    # METRICS
                    # -----------------------------------------

                    recall = (
                        tp / (tp + fn)
                        if (tp + fn) > 0
                        else 0
                    )

                    precision = (
                        tp / (tp + fp)
                        if (tp + fp) > 0
                        else 0
                    )


                    # -----------------------------------------
                    # CONFUSION MATRIX
                    # -----------------------------------------

                    st.subheader(
                        "Confusion Matrix"
                    )

                    c1, c2, c3, c4 = st.columns(4)

                    with c1:
                        st.metric(
                            "True Positives",
                            tp
                        )

                    with c2:
                        st.metric(
                            "False Positives",
                            fp
                        )

                    with c3:
                        st.metric(
                            "False Negatives",
                            fn
                        )

                    with c4:
                        st.metric(
                            "True Negatives",
                            tn
                        )


                    # -----------------------------------------
                    # COST
                    # -----------------------------------------

                    st.subheader(
                        "💰 Cost Analysis"
                    )

                    cost1, cost2, cost3 = st.columns(3)

                    with cost1:

                        st.metric(
                            "Total Cost",
                            f"₹{total_cost:,}"
                        )

                    with cost2:

                        st.metric(
                            "False Positive Cost",
                            f"₹{fp_cost:,}"
                        )

                    with cost3:

                        st.metric(
                            "False Negative Cost",
                            f"₹{fn_cost:,}"
                        )


                    # -----------------------------------------
                    # PERFORMANCE
                    # -----------------------------------------

                    st.subheader(
                        "📈 Performance"
                    )

                    metric1, metric2 = st.columns(2)

                    with metric1:

                        st.metric(
                            "Recall",
                            f"{recall:.2%}"
                        )

                    with metric2:

                        st.metric(
                            "Precision",
                            f"{precision:.2%}"
                        )


            # =================================================
            # PREDICTION MODE
            # =================================================

            else:

                st.header("🚛 Prediction Results")


                if predicted_failures > 0:

                    st.error(
                        f"""
                        🚨 **APS Failure Risk Detected**

                        {predicted_failures:,} truck(s) crossed
                        the cost-sensitive decision threshold
                        of **0.024**.
                        """
                    )

                else:

                    st.success(
                        """
                        ✅ **No APS Failure Detected**

                        No uploaded truck crossed the
                        cost-sensitive decision threshold.
                        """
                    )


            # =================================================
            # DETAILED RESULTS
            # =================================================

            st.header("🔎 Detailed Predictions")


            result_df = pd.DataFrame({

                "Truck": range(
                    1,
                    total_trucks + 1
                ),

                "Failure Probability": probabilities,

                "Prediction": predictions

            })


            result_df["Prediction"] = (
                result_df["Prediction"]
                .map({
                    0: "🟢 Not APS Failure",
                    1: "🔴 APS Failure"
                })
            )


            result_df["Failure Probability"] = (
                result_df["Failure Probability"]
                .map(
                    lambda x: f"{x:.2%}"
                )
            )


            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True
            )


            # =================================================
            # DOWNLOAD
            # =================================================

            st.header("⬇️ Export Results")


            download_df = pd.DataFrame({

                "Truck": range(
                    1,
                    total_trucks + 1
                ),

                "Failure_Probability": probabilities,

                "Prediction": predictions

            })


            csv = download_df.to_csv(
                index=False
            ).encode("utf-8")


            st.download_button(
                label="⬇️ Download Predictions",
                data=csv,
                file_name="scaniaguard_predictions.csv",
                mime="text/csv"
            )


        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )