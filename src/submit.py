
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())
try:
    from src.features import load_data, engineer_features
    from src.train import get_X_y, CHANGE_TYPE_MAP
except ImportError:
    # If running from src
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.features import load_data, engineer_features
    from src.train import get_X_y, CHANGE_TYPE_MAP

def generate_submission():
    print("Loading full data for submission...")
    train_geo, test_geo = load_data('dados/train.geojson', 'dados/test.geojson')
    
    print("Engineering features...")
    train_df = engineer_features(train_geo)
    test_df = engineer_features(test_geo)
    
    X_train, y_train = get_X_y(train_df)
    
    # Handle Test Data
    if 'index' in test_df.columns:
        test_ids = test_df['index']
        # Also drop index for prediction
        X_test = test_df.drop(columns=['index'], errors='ignore')
    else:
        test_ids = test_df.index
        # If no index column, assume index is the ID
        X_test = test_df.copy()
        
    # Ensure columns match (One-Hot Encoding might produce different columns)
    train_cols = X_train.columns
    # Add missing
    for col in train_cols:
        if col not in X_test.columns:
            X_test[col] = 0
            
    # Align
    X_test = X_test[train_cols]
    
    # Train XGBoost
    print(f"Training XGBoost on full data ({X_train.shape[0]} samples)...")
    xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=10, 
                        subsample=1.0, colsample_bytree=1.0, 
                        random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    
    # Train Random Forest (Balanced)
    print("Training Random Forest (Balanced)...")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    print("Predicting test set (Ensemble)...")
    # Soft Voting
    probs_xgb = xgb.predict_proba(X_test)
    probs_rf = rf.predict_proba(X_test)
    
    # Weighted average (give more weight to XGB if it was better, or equal)
    # XGB was 0.46, RF was 0.44. Let's do 0.6 XGB + 0.4 RF
    avg_probs = 0.6 * probs_xgb + 0.4 * probs_rf
    preds = np.argmax(avg_probs, axis=1) # Get class with max probability
    
    # Kaggle sample submission uses integers (0, 1, 2, etc.)
    
    # Kaggle sample submission uses integers (0, 1, 2, etc.)
    # So we don't need to map back to strings!
    pred_labels = preds
    
    submission = pd.DataFrame({
        'Id': test_ids,
        'change_type': pred_labels
    })
    
    submission.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")

if __name__ == "__main__":
    generate_submission()
