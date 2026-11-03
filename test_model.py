
import json
import joblib
import numpy as np
import tensorflow as tf

# Load saved model
model = tf.keras.models.load_model(
    "models/autoencoder.keras"
)

# Load saved scaler
scaler = joblib.load("models/scaler.pkl")

# Load thresholds
with open("models/thresholds.json", "r") as f:
    thresholds = json.load(f)

# Sample sensor reading
sample = np.array([[65.2, 2.1, 30.5, 1500]])

# Normalize using saved scaler
sample_scaled = scaler.transform(sample)

# Reconstruct using trained model
reconstructed = model.predict(sample_scaled, verbose=0)

# Calculate reconstruction error
error = np.mean(
    (sample_scaled - reconstructed) ** 2,
    axis=1
)[0]

threshold = thresholds["percentile"]

print("\nMODEL LOADED SUCCESSFULLY")
print("Original sensor values:", sample)
print("Reconstruction error:", round(float(error), 6))
print("Threshold:", round(threshold, 6))

if error > threshold:
    print("RESULT: ANOMALY DETECTED")
else:
    print("RESULT: NORMAL")
