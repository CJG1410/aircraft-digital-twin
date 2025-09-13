import pandas as pd
import numpy as np

print("Generating synthetic engine data...")

# Set up parameters for a single engine's lifecycle
NUM_CYCLES = 200
NUM_SENSORS = 5

# Generate synthetic sensor data
# We'll model sensor readings as a base value plus noise,
# with a trend that changes as the engine degrades
np.random.seed(42)
sensor_data = pd.DataFrame()

for i in range(1, NUM_SENSORS + 1):
    base_values = np.linspace(50 + i, 80 - i, NUM_CYCLES)
    noise = np.random.normal(0, 1.5, NUM_CYCLES)
    sensor_data[f'sensor_{i}'] = base_values + noise

# Create a feature that represents degradation over time
degradation_factor = np.linspace(0, 1, NUM_CYCLES) ** 2

# Add a subtle, non-linear degradation to sensor readings
for i in range(1, NUM_SENSORS + 1):
    degradation = degradation_factor * (np.random.rand() * 10)
    sensor_data[f'sensor_{i}'] += degradation

# Add RUL (Remaining Useful Life) column
# RUL decreases from NUM_CYCLES to 0
sensor_data['RUL'] = NUM_CYCLES - np.arange(NUM_CYCLES)

# Introduce a few anomalies (outliers) in the data for our anomaly detection model
# We'll randomly spike a few sensor values
anomaly_indices = np.random.choice(range(NUM_CYCLES), size=5, replace=False)
for idx in anomaly_indices:
    for i in range(1, NUM_SENSORS + 1):
        if np.random.rand() > 0.5:
            sensor_data.at[idx, f'sensor_{i}'] *= (1.5 + np.random.rand())

# Save the data
sensor_data.to_csv('data/engine_data.csv', index=False)

print("Synthetic data generated and saved to data/engine_data.csv")