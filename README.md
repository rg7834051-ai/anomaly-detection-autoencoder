# Anomaly Detection in Machine Sensor Readings (Autoencoder)

An autoencoder is trained only on normal synthetic sensor data (temperature, vibration, pressure, RPM). Reconstruction error is used as the anomaly score, and the threshold is chosen on validation data.

## Run
pip install -r requirements.txt
python main.py

## Results (test set)
- 99th percentile threshold: precision 0.941, recall 1.000, F1 0.969
- Best-F1 threshold: precision 1.000, recall 1.000, F1 1.000