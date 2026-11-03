
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score
)

warnings.filterwarnings("ignore")

# =========================================================
# 1. APP CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Sensor AI | Industrial Intelligence",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "autoencoder.keras"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
THRESHOLD_PATH = MODEL_DIR / "thresholds.json"
METADATA_PATH = MODEL_DIR / "metadata.json"
RESULTS_PATH = BASE_DIR / "test_results.csv"

FEATURES = [
    "temperature",
    "vibration",
    "pressure",
    "rpm"
]

# =========================================================
# 2. MIDNIGHT TEAL THEME
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #071B24;
    color: #E6F4F1;
}

[data-testid="stHeader"] {
    background-color: #071B24;
}

[data-testid="stSidebar"] {
    background-color: #0D2934;
    border-right: 1px solid #20515C;
}

[data-testid="stSidebar"] * {
    color: #D5EAE6;
}

h1, h2, h3 {
    color: #F0FDFA !important;
}

p, label {
    color: #C5DAD8;
}

.main-title {
    font-size: 34px;
    font-weight: 750;
    color: #F0FDFA;
}

.subtitle {
    font-size: 15px;
    color: #8EAFB1;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #123744, #102F3B);
    border: 1px solid #20515C;
    padding: 18px;
    border-radius: 14px;
}

[data-testid="stMetricValue"] {
    color: #20C9B0 !important;
    font-weight: 700;
}

[data-testid="stMetricLabel"] {
    color: #B8D3D1 !important;
}

.stButton > button {
    background: linear-gradient(90deg, #0F9F91, #20C9B0);
    color: #062B2B;
    border: none;
    border-radius: 9px;
    font-weight: 700;
}

.stButton > button:hover {
    background: #5EEAD4;
    color: #062B2B;
}

.stDownloadButton > button {
    background-color: #123744;
    color: #5EEAD4;
    border: 1px solid #20C9B0;
    border-radius: 9px;
}

.stTextInput input,
.stNumberInput input {
    background-color: #123744;
    color: #F0FDFA;
}

[data-testid="stDataFrame"] {
    border: 1px solid #20515C;
    border-radius: 12px;
}

[data-testid="stFileUploader"] {
    background-color: #123744;
    border: 1px dashed #20C9B0;
    border-radius: 12px;
    padding: 12px;
}

hr {
    border-color: #20515C;
}

.status-normal {
    background-color: #103D36;
    color: #6EE7B7;
    padding: 16px;
    border-radius: 12px;
    border-left: 5px solid #34D399;
    font-weight: bold;
}

.status-alert {
    background-color: #482631;
    color: #FDA4AF;
    padding: 16px;
    border-radius: 12px;
    border-left: 5px solid #FB7185;
    font-weight: bold;
}

.status-warning {
    background-color: #44391D;
    color: #FDE68A;
    padding: 16px;
    border-radius: 12px;
    border-left: 5px solid #FBBF24;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD SAVED MODEL, SCALER AND THRESHOLDS
# =========================================================

@st.cache_resource
def load_model_files():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "models/autoencoder.keras file not found."
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            "models/scaler.pkl file not found."
        )

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            "models/thresholds.json file not found."
        )

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    scaler = joblib.load(SCALER_PATH)

    with open(THRESHOLD_PATH, "r") as file:
        thresholds = json.load(file)

    return model, scaler, thresholds


try:
    model, scaler, thresholds = load_model_files()
    MODEL_LOADED = True

except Exception as error:
    MODEL_LOADED = False
    st.error(f"Model loading failed: {error}")
    st.stop()


# =========================================================
# 4. PREDICTION FUNCTIONS
# =========================================================

def predict_readings(data, threshold=None):

    if threshold is None:
        threshold = float(thresholds["percentile"])

    X = data[FEATURES].astype(float).values

    X_scaled = scaler.transform(X)

    reconstructed = model.predict(
        X_scaled,
        verbose=0
    )

    errors = np.mean(
        (X_scaled - reconstructed) ** 2,
        axis=1
    )

    predictions = (errors > threshold).astype(int)

    result = data.copy()

    result["reconstruction_error"] = errors
    result["prediction"] = predictions

    result["status"] = np.where(
        predictions == 1,
        "ANOMALY",
        "NORMAL"
    )

    return result


# =========================================================
# 5. LOAD PROJECT DATA
# =========================================================

@st.cache_data
def load_project_data():

    if not RESULTS_PATH.exists():
        return None

    return pd.read_csv(RESULTS_PATH)


try:
    project_data = load_project_data()

    if project_data is None:
        st.error("test_results.csv not found in project folder.")
        st.stop()

    missing_features = [
        col for col in FEATURES
        if col not in project_data.columns
    ]

    if missing_features:
        st.error(
            f"Missing sensor columns in test_results.csv: "
            f"{missing_features}"
        )
        st.stop()

    baseline_data = predict_readings(project_data)

except Exception as error:
    st.error(f"Could not process project data: {error}")
    st.stop()


# =========================================================
# 6. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown(
        "<h1 style='color:#20C9B0;'>⚙ SENSOR AI</h1>",
        unsafe_allow_html=True
    )

    st.caption("Industrial Anomaly Monitoring System")

    st.divider()

    st.markdown("### NAVIGATION")

    page = st.radio(
        "Select Dashboard Page",
        [
            "Overview",
            "Sensor Monitoring",
            "Anomaly Detection",
            "Alert Center",
            "Model Performance",
            "Data Upload",
            "Settings"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### MODEL INFORMATION")

    st.success("✓ Autoencoder Loaded")

    st.caption("Engine: TensorFlow / Keras")
    st.caption("Input Sensors: 4")

    st.markdown("### Detection Threshold")

    threshold_choice = st.selectbox(
        "Choose threshold",
        ["99th Percentile", "Best F1"],
        label_visibility="collapsed"
    )

    if threshold_choice == "99th Percentile":
        ACTIVE_THRESHOLD = float(thresholds["percentile"])
    else:
        ACTIVE_THRESHOLD = float(thresholds["best_f1"])

    st.caption(
        f"Current threshold: {ACTIVE_THRESHOLD:.6f}"
    )

    st.divider()

    st.caption("Sensor AI | Autoencoder Prototype")


# =========================================================
# 7. COMMON HEADER
# =========================================================

st.markdown(
    '<div class="main-title">Industrial Sensor Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered sensor monitoring and anomaly detection'
    '</div>',
    unsafe_allow_html=True
)

st.caption(
    "Data source: Project test_results.csv | "
    "Model: Saved TensorFlow Autoencoder"
)

st.divider()


# =========================================================
# 8. COMMON CHART FUNCTIONS
# =========================================================

def show_sensor_chart(data, sensor):

    fig, ax = plt.subplots(figsize=(11, 4))

    fig.patch.set_facecolor("#071B24")
    ax.set_facecolor("#102F3B")

    ax.plot(
        data[sensor].values,
        color="#20C9B0",
        linewidth=1.5,
        label=sensor.capitalize()
    )

    if "prediction" in data.columns:

        flagged = data["prediction"].values == 1
        indices = np.where(flagged)[0]

        if len(indices) > 0:

            ax.scatter(
                indices,
                data[sensor].values[indices],
                color="#FB7185",
                s=25,
                label="Detected Anomaly",
                zorder=3
            )

    ax.set_title(
        f"{sensor.capitalize()} Monitoring",
        color="#F0FDFA",
        fontsize=15
    )

    ax.set_xlabel("Reading Number", color="#B8D3D1")
    ax.set_ylabel(sensor.capitalize(), color="#B8D3D1")

    ax.tick_params(colors="#B8D3D1")
    ax.grid(color="#42616A", alpha=0.3, linestyle="--")

    for spine in ax.spines.values():
        spine.set_color("#42616A")

    legend = ax.legend(
        facecolor="#123744",
        edgecolor="#20515C"
    )

    for item in legend.get_texts():
        item.set_color("#F0FDFA")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_error_chart(data, threshold):

    fig, ax = plt.subplots(figsize=(11, 4))

    fig.patch.set_facecolor("#071B24")
    ax.set_facecolor("#102F3B")

    ax.plot(
        data["reconstruction_error"].values,
        color="#20C9B0",
        linewidth=1.4,
        label="Reconstruction Error"
    )

    ax.axhline(
        threshold,
        color="#FBBF24",
        linestyle="--",
        linewidth=2,
        label="Detection Threshold"
    )

    flagged = (
        data["reconstruction_error"].values > threshold
    )

    indices = np.where(flagged)[0]

    if len(indices) > 0:
        ax.scatter(
            indices,
            data["reconstruction_error"].values[indices],
            color="#FB7185",
            s=22,
            label="Detected Anomaly"
        )

    ax.set_title(
        "Anomaly Detection Score",
        color="#F0FDFA",
        fontsize=15
    )

    ax.set_xlabel("Reading Number", color="#B8D3D1")
    ax.set_ylabel("Reconstruction Error", color="#B8D3D1")

    ax.tick_params(colors="#B8D3D1")
    ax.grid(color="#42616A", alpha=0.3, linestyle="--")

    for spine in ax.spines.values():
        spine.set_color("#42616A")

    legend = ax.legend(
        facecolor="#123744",
        edgecolor="#20515C"
    )

    for item in legend.get_texts():
        item.set_color("#F0FDFA")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(5, 4))

    fig.patch.set_facecolor("#071B24")
    ax.set_facecolor("#102F3B")

    image = ax.imshow(
        cm,
        cmap="GnBu"
    )

    ax.set_title(
        "Confusion Matrix",
        color="white"
    )

    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("Actual", color="white")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Normal", "Anomaly"],
        color="white"
    )

    ax.set_yticklabels(
        ["Normal", "Anomaly"],
        color="white"
    )

    for i in range(2):
        for j in range(2):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white",
                fontsize=15,
                fontweight="bold"
            )

    fig.colorbar(image, ax=ax)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# =========================================================
# 9. OVERVIEW PAGE
# =========================================================

if page == "Overview":

    st.subheader("System Overview")

    total = len(baseline_data)

    anomaly_count = int(
        baseline_data["prediction"].sum()
    )

    normal_count = total - anomaly_count

    anomaly_rate = (
        anomaly_count / total * 100
        if total > 0 else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Readings", f"{total:,}")
    c2.metric("Normal Readings", f"{normal_count:,}")
    c3.metric("Anomalies Detected", f"{anomaly_count:,}")
    c4.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")

    st.divider()

    if anomaly_count > 0:

        st.markdown(
            f'<div class="status-alert">'
            f'⚠ {anomaly_count} anomalous readings detected. '
            f'Please review the Alert Center.'
            f'</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="status-normal">'
            '✓ No anomalous readings detected.'
            '</div>',
            unsafe_allow_html=True
        )

    st.subheader("Sensor Readings")

    selected_sensor = st.selectbox(
        "Choose sensor",
        FEATURES
    )

    show_sensor_chart(
        baseline_data,
        selected_sensor
    )

    st.subheader("Anomaly Detection Score")

    show_error_chart(
        baseline_data,
        ACTIVE_THRESHOLD
    )

    st.caption(
        "This overview uses the project's saved test dataset. "
        "It is not a live hardware sensor stream."
    )


# =========================================================
# 10. SENSOR MONITORING PAGE
# =========================================================

elif page == "Sensor Monitoring":

    st.subheader("Sensor Monitoring")

    st.write(
        "Monitor the readings of all four machine sensors."
    )

    for sensor in FEATURES:

        st.markdown(f"### {sensor.capitalize()}")

        show_sensor_chart(
            baseline_data,
            sensor
        )

        st.divider()

    st.subheader("Sensor Data Preview")

    st.dataframe(
        baseline_data[
            FEATURES + [
                "reconstruction_error",
                "status"
            ]
        ].head(20),
        use_container_width=True
    )


# =========================================================
# 11. ANOMALY DETECTION PAGE
# =========================================================

elif page == "Anomaly Detection":

    st.subheader("Anomaly Detection Analysis")

    show_error_chart(
        baseline_data,
        ACTIVE_THRESHOLD
    )

    st.divider()

    st.subheader("Manual Sensor Reading Test")

    st.write(
        "Enter sensor values to check whether a reading "
        "is normal or anomalous."
    )

    with st.form("manual_sensor_form"):

        c1, c2 = st.columns(2)

        with c1:

            temperature = st.number_input(
                "Temperature",
                value=65.02,
                format="%.2f"
            )

            pressure = st.number_input(
                "Pressure",
                value=30.05,
                format="%.2f"
            )

        with c2:

            vibration = st.number_input(
                "Vibration",
                value=2.01,
                format="%.2f"
            )

            rpm = st.number_input(
                "RPM",
                value=1500.00,
                format="%.2f"
            )

        analyze = st.form_submit_button(
            "Analyze Sensor Reading",
            use_container_width=True
        )

    if analyze:

        manual_data = pd.DataFrame(
            [[temperature, vibration, pressure, rpm]],
            columns=FEATURES
        )

        prediction_result = predict_readings(
            manual_data,
            ACTIVE_THRESHOLD
        )

        error_value = float(
            prediction_result["reconstruction_error"].iloc[0]
        )

        prediction = int(
            prediction_result["prediction"].iloc[0]
        )

        st.divider()

        st.metric(
            "Reconstruction Error",
            f"{error_value:.6f}"
        )

        if prediction == 1:

            st.markdown(
                '<div class="status-alert">'
                '⚠ ANOMALY DETECTED'
                '</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="status-normal">'
                '✓ NORMAL READING'
                '</div>',
                unsafe_allow_html=True
            )

        st.write(
            f"Detection threshold: {ACTIVE_THRESHOLD:.6f}"
        )

        st.dataframe(
            prediction_result[
                FEATURES + [
                    "reconstruction_error",
                    "status"
                ]
            ],
            use_container_width=True
        )


# =========================================================
# 12. ALERT CENTER PAGE
# =========================================================

elif page == "Alert Center":

    st.subheader("Alert Center")

    alerts = baseline_data[
        baseline_data["prediction"] == 1
    ].copy()

    alert_count = len(alerts)

    st.metric(
        "Total Anomaly Alerts",
        alert_count
    )

    if alert_count == 0:

        st.success("No anomalies detected.")

    else:

        st.warning(
            f"{alert_count} readings require review."
        )

        alerts.insert(
            0,
            "Reading Number",
            alerts.index + 1
        )

        st.dataframe(
            alerts[
                ["Reading Number"]
                + FEATURES
                + [
                    "reconstruction_error",
                    "status"
                ]
            ],
            use_container_width=True
        )

        csv_data = alerts.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Alert Report",
            data=csv_data,
            file_name="sensor_anomaly_alerts.csv",
            mime="text/csv"
        )

    st.caption(
        "Alerts are displayed in this dashboard only. "
        "They do not trigger email or hardware notifications."
    )


# =========================================================
# 13. MODEL PERFORMANCE PAGE
# =========================================================

elif page == "Model Performance":

    st.subheader("Model Performance")

    if "label" not in baseline_data.columns:

        st.warning(
            "Ground-truth labels are unavailable. "
            "Evaluation metrics cannot be calculated."
        )

    else:

        y_true = baseline_data["label"].astype(int).values
        y_pred = baseline_data["prediction"].astype(int).values

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Accuracy", f"{accuracy:.3f}")
        c2.metric("Precision", f"{precision:.3f}")
        c3.metric("Recall", f"{recall:.3f}")
        c4.metric("F1 Score", f"{f1:.3f}")

        st.divider()

        st.subheader("Confusion Matrix")

        show_confusion_matrix(
            y_true,
            y_pred
        )

    st.subheader("Reconstruction Error Distribution")

    fig, ax = plt.subplots(figsize=(10, 4))

    fig.patch.set_facecolor("#071B24")
    ax.set_facecolor("#102F3B")

    ax.hist(
        baseline_data["reconstruction_error"],
        bins=40,
        color="#20C9B0",
        alpha=0.8
    )

    ax.axvline(
        ACTIVE_THRESHOLD,
        color="#FBBF24",
        linestyle="--",
        linewidth=2,
        label="Threshold"
    )

    ax.set_title(
        "Reconstruction Error Distribution",
        color="white"
    )

    ax.set_xlabel("Reconstruction Error", color="white")
    ax.set_ylabel("Frequency", color="white")
    ax.tick_params(colors="white")
    ax.legend()

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Evaluation is based on the project's synthetic test dataset. "
        "These results do not establish real-world industrial accuracy."
    )


# =========================================================
# 14. DATA UPLOAD PAGE
# =========================================================

elif page == "Data Upload":

    st.subheader("Upload Sensor Dataset")

    st.write(
        "Upload a CSV containing temperature, vibration, "
        "pressure and rpm columns."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            uploaded_data = pd.read_csv(
                uploaded_file
            )

            missing = [
                col for col in FEATURES
                if col not in uploaded_data.columns
            ]

            if missing:

                st.error(
                    f"Missing required columns: {missing}"
                )

            elif uploaded_data.empty:

                st.error("The uploaded CSV is empty.")

            else:

                numeric_data = uploaded_data[
                    FEATURES
                ].apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                if numeric_data.isnull().any().any():

                    st.error(
                        "Sensor columns contain missing or "
                        "non-numeric values. Please correct the CSV."
                    )

                elif not np.isfinite(
                    numeric_data.to_numpy(dtype=float)
                ).all():

                    st.error(
                        "Sensor data contains infinite values."
                    )

                else:

                    st.success(
                        "CSV uploaded and validated successfully."
                    )

                    st.subheader("Uploaded Data Preview")

                    st.dataframe(
                        uploaded_data.head(20),
                        use_container_width=True
                    )

                    uploaded_result = predict_readings(
                        uploaded_data,
                        ACTIVE_THRESHOLD
                    )

                    total = len(uploaded_result)

                    anomaly_count = int(
                        uploaded_result["prediction"].sum()
                    )

                    normal_count = total - anomaly_count

                    c1, c2, c3 = st.columns(3)

                    c1.metric("Total Readings", total)
                    c2.metric("Normal", normal_count)
                    c3.metric("Anomalies", anomaly_count)

                    st.subheader("Prediction Results")

                    st.dataframe(
                        uploaded_result[
                            FEATURES + [
                                "reconstruction_error",
                                "status"
                            ]
                        ],
                        use_container_width=True
                    )

                    st.download_button(
                        "Download Prediction Results",
                        data=uploaded_result.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="sensor_prediction_results.csv",
                        mime="text/csv"
                    )

                    st.subheader("Uploaded Data Anomaly Graph")

                    show_error_chart(
                        uploaded_result,
                        ACTIVE_THRESHOLD
                    )

        except Exception as error:

            st.error(
                f"Unable to process uploaded CSV: {error}"
            )


# =========================================================
# 15. SETTINGS PAGE
# =========================================================

elif page == "Settings":

    st.subheader("System Settings")

    st.markdown("### Model Information")

    st.success("TensorFlow Autoencoder: Loaded")

    st.write(
        f"Model file: {MODEL_PATH.name}"
    )

    st.write(
        f"Scaler file: {SCALER_PATH.name}"
    )

    st.write(
        f"Threshold file: {THRESHOLD_PATH.name}"
    )

    st.write(
        f"Input features: {', '.join(FEATURES)}"
    )

    st.divider()

    st.markdown("### Detection Thresholds")

    st.write(
        f"99th Percentile: "
        f"{float(thresholds['percentile']):.6f}"
    )

    st.write(
        f"Best F1 Threshold: "
        f"{float(thresholds['best_f1']):.6f}"
    )

    st.write(
        f"Active Threshold: {ACTIVE_THRESHOLD:.6f}"
    )

    st.divider()

    st.markdown("### Project Information")

    st.write("Project: Anomaly Detection in Machine Sensor Readings")
    st.write("Algorithm: Autoencoder Neural Network")
    st.write("Framework: TensorFlow / Keras")
    st.write("Preprocessing: StandardScaler")
    st.write("Detection: Reconstruction Error Thresholding")

    st.info(
        "This prototype currently uses saved synthetic test data "
        "and uploaded CSV files. Hardware integration and "
        "real-time sensor streaming are future extensions."
    )

    if METADATA_PATH.exists():

        st.markdown("### Model Metadata")

        try:

            with open(METADATA_PATH, "r") as file:
                metadata = json.load(file)

            st.json(metadata)

        except Exception:
            st.warning("Could not read model metadata.")


# =========================================================
# 16. FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#8EAFB1;
        padding:15px;
        font-size:13px;
    ">
        Industrial Sensor Intelligence |
        Autoencoder-based Anomaly Detection |
        Midnight Teal Edition
    </div>
    """,
    unsafe_allow_html=True
)
