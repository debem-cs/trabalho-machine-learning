
import pandas as pd
import numpy as np
import joblib
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())
try:
    from src.features import load_data, engineer_features
    from src.train import get_X_y, CHANGE_TYPE_MAP
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.features import load_data, engineer_features
    from src.train import get_X_y, CHANGE_TYPE_MAP

def run_inference(model_path='model.joblib', test_path='dados/test.geojson'):
    print(f"Loading model from {model_path}...")
    try:
        model_data = joblib.load(model_path)
    except FileNotFoundError:
        print("Model file not found. Please run src/submit.py first to train and save the model.")
        return

    xgb = model_data['xgb']
    rf = model_data['rf']
    train_cols = model_data['features']
    
    print(f"Loading test data from {test_path}...")
    # Load test data (just a few rows to demonstrate)
    test_df = list(load_data('dados/train.geojson', test_path, rows=slice(0, 10)))[1] # Hack to get test df using existing load_data structure which returns tuple
    
    # Or better, just use read_file directly if we want new data
    # But let's stick to our pipeline
    _, test_geo = load_data('dados/train.geojson', test_path)

    print("Engineering features...")
    test_df = engineer_features(test_geo)
    
    # Prepare Test Data
    if 'index' in test_df.columns:
        test_ids = test_df['index']
        X_test = test_df.drop(columns=['index'], errors='ignore')
    else:
        test_ids = test_df.index
        X_test = test_df.copy()
        
    # Align Columns
    for col in train_cols:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[train_cols]
    
    print("Predicting (Ensemble)...")
    probs_xgb = xgb.predict_proba(X_test)
    probs_rf = rf.predict_proba(X_test)
    
    avg_probs = 0.6 * probs_xgb + 0.4 * probs_rf
    preds = np.argmax(avg_probs, axis=1)
    
    print("Predictions:", preds[:10])
    print("Done.")

if __name__ == "__main__":
    run_inference()
