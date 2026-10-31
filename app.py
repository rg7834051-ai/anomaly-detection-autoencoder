import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")
HAS_TF = False   # live training uses the light scikit-learn autoencoder

st.set_page_config(page_title="Sensor Anomaly Detection", layout="wide")
st.title("Anomaly Detection in Machine Sensor Readings")
st.write("An autoencoder is trained only on normal sensor data. Readings with high "
         "reconstruction error are flagged as anomalies.")


# ---------------- helpers ----------------
def show_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.3f}")
    c2.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.3f}")
    c3.metric("F1-score", f"{f1_score(y_true, y_pred, zero_division=0):.3f}")
    c4.metric("False alarms (FP)", int(fp))
    c5.metric("Missed anomalies (FN)", int(fn))


def plot_scores(err, threshold):
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(err, label="Reconstruction error")
    ax.axhline(threshold, color="red", linestyle="--", label="Threshold")
    ax.set_yscale("log")
    ax.set_xlabel("Test time step")
    ax.set_ylabel("Error (log scale)")
    ax.legend()
    st.pyplot(fig)


def plot_flagged(values, y_true, y_pred, name):
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(values, alpha=0.7, label=name)
    flag = np.where(y_pred == 1)[0]
    true_idx = np.where(y_true == 1)[0]
    ax.scatter(flag, values[flag], color="red", s=15, label="Flagged")
    ax.scatter(true_idx, values[true_idx], facecolors="none",
               edgecolors="black", s=40, label="True anomaly")
    ax.legend()
    st.pyplot(fig)


@st.cache_data
def make_data():
    np.random.seed(42)
    N = 6000
    t = np.arange(N)
    cycle = np.sin(2 * np.pi * t / 200)
    df = pd.DataFrame({
        "temperature": 60 + 5*cycle   + np.random.normal(0, 0.5, N),
        "vibration":   2  + 0.5*cycle + np.random.normal(0, 0.1, N),
        "pressure":    30 + 3*cycle   + np.random.normal(0, 0.4, N),
        "rpm":         1500 + 50*cycle + np.random.normal(0, 5, N),
    })
    df["label"] = 0

    def inject(df, start, end, n_events, seed):
        rng = np.random.default_rng(seed)
        starts = rng.choice(np.arange(start, end - 10), n_events, replace=False)
        for s in starts:
            e = s + rng.integers(3, 8)
            if rng.random() < 0.5:
                df.loc[s:e, "temperature"] += 12
                df.loc[s:e, "vibration"] += 1.5
            else:
                df.loc[s:e, "pressure"] -= 8
                df.loc[s:e, "rpm"] -= 150
            df.loc[s:e, "label"] = 1
        return df

    df = inject(df, 3600, 4800, 12, 1)
    df = inject(df, 4800, 6000, 15, 2)
    return df


def train(X_train, X_val_normal, bottleneck, epochs, seed):
    if HAS_TF:
        keras.utils.set_random_seed(seed)
        model = keras.Sequential([
            layers.Input(shape=(4,)),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(bottleneck, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(4, activation="linear"),
        ])
        model.compile(optimizer="adam", loss="mse")
        es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                           restore_best_weights=True)
        hist = model.fit(X_train, X_train, epochs=epochs, batch_size=32,
                         validation_data=(X_val_normal, X_val_normal),
                         callbacks=[es], verbose=0)
        return (lambda X: model.predict(X, verbose=0)), hist.history["loss"]
    mlp = MLPRegressor(hidden_layer_sizes=(16, 8, bottleneck, 8, 16),
                       activation="relu", max_iter=epochs, random_state=seed)
    mlp.fit(X_train, X_train)
    return mlp.predict, mlp.loss_curve_


def recon_error(predict, X):
    return np.mean((X - predict(X)) ** 2, axis=1)


tab1, tab2 = st.tabs(["Report results", "Live training"])

# ================= TAB 1: report results (saved Keras run) =================
with tab1:
    st.caption("Results of the Keras autoencoder run reported in the project report.")
    try:
        saved = pd.read_csv("test_results.csv")
        th = json.load(open("thresholds.json"))
    except FileNotFoundError:
        st.error("test_results.csv or thresholds.json not found in the repository.")
    else:
        left, right = st.columns(2)
        choice = left.radio("Threshold", ["99th percentile", "Best F1", "Custom"],
                            horizontal=True, key="s_choice")
        sensor = right.selectbox("Sensor to plot",
                                 ["temperature", "vibration", "pressure", "rpm"],
                                 key="s_sensor")
        if choice == "99th percentile":
            threshold = th["percentile"]
        elif choice == "Best F1":
            threshold = th["best_f1"]
        else:
            exp = st.slider("Custom threshold (log10 scale)", -3.0, 1.5, -1.0, 0.05,
                            key="s_custom")
            threshold = 10 ** exp
        st.write(f"Threshold = **{threshold:.4f}**")

        y_true = saved["label"].values
        y_pred = (saved["error"].values > threshold).astype(int)
        show_metrics(y_true, y_pred)

        st.subheader("Anomaly score over the test period")
        plot_scores(saved["error"].values, threshold)

        st.subheader(f"Flagged points on the {sensor} sensor")
        plot_flagged(saved[sensor].values, y_true, y_pred, sensor)

# ================= TAB 2: live training =================
with tab2:
    engine = "Keras / TensorFlow" if HAS_TF else "scikit-learn MLP autoencoder"
    st.caption(f"Engine: {engine}. The model is trained here, on normal data only.")

    c1, c2, c3 = st.columns(3)
    epochs = c1.slider("Epochs", 5, 100, 60, 5, key="l_epochs")
    bottleneck = c2.slider("Bottleneck size", 2, 4, 3, key="l_bottleneck")
    seed = c3.number_input("Random seed", 0, 999, 42, key="l_seed")

    df = make_data()
    features = ["temperature", "vibration", "pressure", "rpm"]
    train_df, val_df, test_df = df.iloc[:3600], df.iloc[3600:4800], df.iloc[4800:]
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    if st.button("Train model", type="primary", key="l_train"):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[features].values)
        X_val = scaler.transform(val_df[features].values)
        X_test = scaler.transform(test_df[features].values)
        with st.spinner("Training the autoencoder on normal data..."):
            predict, loss_curve = train(X_train, X_val[y_val == 0],
                                        bottleneck, epochs, int(seed))
        st.session_state["res"] = {
            "err_val": recon_error(predict, X_val),
            "err_test": recon_error(predict, X_test),
            "loss": list(loss_curve),
        }

    if "res" not in st.session_state:
        st.info("Choose the model options and press **Train model**.")
    else:
        res = st.session_state["res"]
        pct = st.slider("Threshold percentile (of normal validation errors)",
                        90.0, 99.9, 99.0, 0.1, key="l_pct")
        threshold = np.percentile(res["err_val"][y_val == 0], pct)
        st.write(f"Threshold = **{threshold:.4f}**")

        y_pred = (res["err_test"] > threshold).astype(int)
        show_metrics(y_test, y_pred)

        st.subheader("Training loss")
        fig0, ax0 = plt.subplots(figsize=(8, 3))
        ax0.plot(res["loss"])
        ax0.set_xlabel("Epoch")
        ax0.set_ylabel("Loss")
        st.pyplot(fig0)

        st.subheader("Anomaly score over the test period")
        plot_scores(res["err_test"], threshold)

        st.subheader("Flagged points on the temperature sensor")
        plot_flagged(test_df["temperature"].values, y_test, y_pred, "temperature")

st.caption("Synthetic, non-sensitive data. Threshold chosen on validation data, "
           "never on test data.")