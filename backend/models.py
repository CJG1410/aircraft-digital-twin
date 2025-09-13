import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, IsolationForest
import joblib

def train_models():
    """Loads synthetic data, trains RUL and Anomaly Detection models, and saves them."""
    print("Loading data and training models...")

    # Load the synthetic data
    try:
        df = pd.read_csv('data/engine_data.csv')
    except FileNotFoundError:
        print("Error: data/engine_data.csv not found. Please run simulate.py first.")
        return

    # --- 1. Train RUL Prediction Model ---
    # Features (sensor data) and target (RUL)
    features = [col for col in df.columns if 'sensor' in col]
    X = df[features]
    y = df['RUL']

    # Split data for training and testing (optional for this simple case but good practice)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest Regressor
    # A tree-based ensemble model that's great for regression tasks
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Save the trained RUL model
    joblib.dump(rf_model, 'data/rul_model.joblib')
    print("RUL prediction model (Random Forest) trained and saved.")

    # --- 2. Train Anomaly Detection Model ---
    # Isolation Forest is an unsupervised model, so we don't need a target variable
    # It works by isolating anomalies (outliers) from normal data
    # The 'contamination' parameter estimates the proportion of outliers in the data
    iso_forest = IsolationForest(random_state=42)
    iso_forest.fit(X)

    # Save the trained Anomaly Detection model
    joblib.dump(iso_forest, 'data/anomaly_model.joblib')
    print("Anomaly detection model (Isolation Forest) trained and saved.")

if __name__ == '__main__':
    train_models()