import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

N = 6000
t = np.arange(N)
cycle = np.sin(2 * np.pi * t / 200)   # machine ka operating cycle

df = pd.DataFrame({
    "temperature": 60 + 5*cycle   + np.random.normal(0, 0.5, N),
    "vibration":   2  + 0.5*cycle + np.random.normal(0, 0.1, N),
    "pressure":    30 + 3*cycle   + np.random.normal(0, 0.4, N),
    "rpm":         1500 + 50*cycle + np.random.normal(0, 5, N),
})
df["label"] = 0   # 0 = normal, 1 = anomaly

print(df.head())
print(df.shape)

# Pehle 600 points ka plot, taaki data dikh jaye
df[["temperature", "vibration", "pressure"]].iloc[:600].plot(figsize=(10, 4), subplots=True)
plt.savefig("plot0_raw_data.png", dpi=150, bbox_inches="tight")
plt.close()
def inject_anomalies(df, start, end, n_events, seed):
    rng = np.random.default_rng(seed)
    starts = rng.choice(np.arange(start, end - 10), n_events, replace=False)
    for s in starts:
        e = s + rng.integers(3, 8)           # 3 se 7 points lamba event
        if rng.random() < 0.5:               # Type 1: overheating + vibration badhna
            df.loc[s:e, "temperature"] += 12
            df.loc[s:e, "vibration"]   += 1.5
        else:                                # Type 2: pressure drop + rpm girna
            df.loc[s:e, "pressure"] -= 8
            df.loc[s:e, "rpm"]      -= 150
        df.loc[s:e, "label"] = 1
    return df

df = inject_anomalies(df, 3600, 4800, n_events=12, seed=1)   # validation period
df = inject_anomalies(df, 4800, 6000, n_events=15, seed=2)   # test period

print("Anomaly points in validation:", df.iloc[3600:4800]["label"].sum())
print("Anomaly points in test:", df.iloc[4800:]["label"].sum())
print("Anomaly points in train:", df.iloc[:3600]["label"].sum())
from sklearn.preprocessing import StandardScaler

features = ["temperature", "vibration", "pressure", "rpm"]

train = df.iloc[:3600]
val   = df.iloc[3600:4800]
test  = df.iloc[4800:]

X_train, y_train = train[features].values, train["label"].values
X_val,   y_val   = val[features].values,   val["label"].values
X_test,  y_test  = test[features].values,  test["label"].values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # fit sirf train par
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

print("Train/Val/Test shapes:", X_train_s.shape, X_val_s.shape, X_test_s.shape)
print("Train mean (should be ~0):", X_train_s.mean(axis=0).round(3))
print("Train std (should be ~1):", X_train_s.std(axis=0).round(3))

# Test period ka plot: anomaly red me dikhegi
plt.figure(figsize=(12, 4))
plt.plot(test["temperature"].values, label="Temperature", alpha=0.7)
idx = np.where(y_test == 1)[0]
plt.scatter(idx, test["temperature"].values[idx], color="red", s=10, label="Anomaly")
plt.legend(); plt.title("Test period with injected anomalies")
plt.savefig("plot0b_test_anomalies.png", dpi=150, bbox_inches="tight")
plt.close()
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

keras.utils.set_random_seed(42)

model = keras.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8,  activation="relu"),
    layers.Dense(3,  activation="relu"),    # bottleneck
    layers.Dense(8,  activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(4,  activation="linear"),  # output = 4 sensor values
])
model.compile(optimizer="adam", loss="mse")
model.summary()

# Validation me sirf normal rows (sirf training monitor karne ke liye)
X_val_normal = X_val_s[y_val == 0]

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True)

history = model.fit(
    X_train_s, X_train_s,          # input aur target same hain
    epochs=100, batch_size=32,
    validation_data=(X_val_normal, X_val_normal),
    callbacks=[early_stop], verbose=0)

print("Stopped at epoch:", len(history.history["loss"]))
print("Best val loss:", min(history.history["val_loss"]))

plt.figure(figsize=(7, 4))
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend(); plt.title("Training Loss")
plt.savefig("plot1_training_loss.png", dpi=150, bbox_inches="tight")
plt.close()
from sklearn.metrics import f1_score

# ---- Step 6: reconstruction error ----
def recon_error(model, X):
    X_pred = model.predict(X, verbose=0)
    return np.mean((X - X_pred) ** 2, axis=1)   # har row ka MSE

err_train = recon_error(model, X_train_s)
err_val   = recon_error(model, X_val_s)
err_test  = recon_error(model, X_test_s)

print("Mean error - train:", err_train.mean())
print("Mean error - val normal:", err_val[y_val == 0].mean())
print("Mean error - val anomaly:", err_val[y_val == 1].mean())

# ---- Step 7: threshold (sirf validation se) ----
# Method A: normal validation errors ka 99th percentile
threshold = np.percentile(err_val[y_val == 0], 99)

# Method B: validation par best F1 dene wala threshold
candidates = np.linspace(err_val.min(), err_val.max(), 200)
f1s = [f1_score(y_val, err_val > th) for th in candidates]
best_f1 = max(f1s)
good = candidates[np.array(f1s) == best_f1]
threshold_f1 = np.sqrt(good.min() * good.max())   # perfect-F1 range ka beech

print("Percentile threshold:", threshold)

print("Best F1 on validation:", max(f1s))

# Histogram (log scale, kyunki anomaly errors bahut bade ho sakte hain)
plt.figure(figsize=(8, 4))
bins = np.logspace(np.log10(err_val.min()), np.log10(err_val.max()), 60)
plt.hist(err_val[y_val == 0], bins=bins, alpha=0.7, label="Normal")
plt.hist(err_val[y_val == 1], bins=bins, alpha=0.7, label="Anomaly")
plt.axvline(threshold, color="red", linestyle="--", label="99th percentile")
plt.axvline(threshold_f1, color="green", linestyle="--", label="Best F1")
plt.xscale("log")
plt.xlabel("Reconstruction error (log scale)")
plt.legend(); plt.title("Threshold Selection (Validation)")
plt.savefig("plot2_threshold.png", dpi=150, bbox_inches="tight")
plt.close()
from sklearn.metrics import confusion_matrix, classification_report

# Best-F1 threshold: perfect-F1 range ka beech (geometric mean)
best_f1 = max(f1s)
good = candidates[np.array(f1s) == best_f1]
threshold_f1 = np.sqrt(good.min() * good.max())
print("Best-F1 threshold (range ka beech):", threshold_f1)

# ---- Test set evaluation (dono thresholds) ----
for name, th in [("Percentile (99th)", threshold), ("Best-F1", threshold_f1)]:
    y_pred_t = (err_test > th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    print("\n=====", name, "| threshold =", round(th, 4), "=====")
    print("TN, FP, FN, TP:", tn, fp, fn, tp)
    print(classification_report(y_test, y_pred_t, target_names=["Normal", "Anomaly"], digits=3))

# Main threshold = percentile wala (plots isi ke hain)
y_pred = (err_test > threshold).astype(int)

# Plot 3: anomaly score over time (test)
plt.figure(figsize=(12, 4))
plt.plot(err_test, label="Reconstruction error")
plt.axhline(threshold, color="red", linestyle="--", label="99th percentile")
plt.axhline(threshold_f1, color="green", linestyle="--", label="Best F1")
plt.yscale("log")
plt.xlabel("Test time step"); plt.ylabel("Error (log scale)")
plt.legend(); plt.title("Anomaly Scores on Test Period")
plt.savefig("plot3_scores.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 4: temperature par flagged points
plt.figure(figsize=(12, 4))
plt.plot(test["temperature"].values, label="Temperature", alpha=0.7)
flag = np.where(y_pred == 1)[0]
plt.scatter(flag, test["temperature"].values[flag], color="red", s=15, label="Flagged")
true_idx = np.where(y_test == 1)[0]
plt.scatter(true_idx, test["temperature"].values[true_idx], facecolors="none",
            edgecolors="black", s=40, label="True anomaly")
plt.legend(); plt.title("Flagged Points on Temperature Sensor")
plt.savefig("plot4_flagged.png", dpi=150, bbox_inches="tight")
plt.close()
import json

out = test.copy().reset_index(drop=True)
out["error"] = err_test
out["label"] = y_test
out.to_csv("test_results.csv", index=False)

with open("thresholds.json", "w") as f:
    json.dump({"percentile": float(threshold), "best_f1": float(threshold_f1)}, f)

print("Saved test_results.csv and thresholds.json")