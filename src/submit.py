
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
    # Using parameters that worked well for Weighted F1 in benchmark (and previous tuning)
    xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=10, 
                        subsample=1.0, colsample_bytree=1.0, 
                        random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    
    # Train Random Forest (Standard - NOT Balanced)
    # Balanced weights hurt Weighted F1 because they penalize majority class errors too much.
    # Standard RF focuses on Accuracy/Weighted F1 naturally.
    print("Training Random Forest (Standard)...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    print("Predicting test set (Ensemble)...")
    # Soft Voting
    probs_xgb = xgb.predict_proba(X_test)
    probs_rf = rf.predict_proba(X_test)
    
    # Weighted average
    # RF Standard (0.747) was slightly better than XGB (0.742) on benchmark.
    # Let's give them equal weight or slightly favor RF.
    # Weighted average
    # RF Standard (0.747) was slightly better than XGB (0.742) on benchmark.
    # Let's give them equal weight or slightly favor RF.
    avg_probs = 0.5 * probs_xgb + 0.5 * probs_rf
    
    # --- PROCEED WITH PSEUDO-LABELING ---
    print("\n--- Starting Pseudo-Labeling ---")
    confidence_threshold = 0.95
    
    # Get max probability for each row
    max_probs = np.max(avg_probs, axis=1)
    # Get predicted label
    initial_preds = np.argmax(avg_probs, axis=1)
    
    # Identify high-confidence samples
    high_conf_idx = np.where(max_probs > confidence_threshold)[0]
    print(f"Found {len(high_conf_idx)} high-confidence test samples (prob > {confidence_threshold}) out of {len(X_test)}")
    
    if len(high_conf_idx) > 0:
        # Create pseudo-labeled data
        X_pseudo = X_test.iloc[high_conf_idx].copy()
        y_pseudo = initial_preds[high_conf_idx]
        
        # Combine with original training data
        print("Augmenting training data with pseudo-labels...")
        X_train_aug = pd.concat([X_train, X_pseudo], axis=0)
        y_train_aug = pd.concat([y_train, pd.Series(y_pseudo)], axis=0)
        
        print(f"New training set size: {X_train_aug.shape[0]} (Original: {X_train.shape[0]})")
        
        # Retrain XGBoost on Augmentation
        print("Retraining XGBoost on augmented data...")
        xgb_aug = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=10, 
                            subsample=1.0, colsample_bytree=1.0, 
                            random_state=42, n_jobs=-1)
        xgb_aug.fit(X_train_aug, y_train_aug)
        
        # Retrain RandomForest on Augmentation
        print("Retraining Random Forest on augmented data...")
        rf_aug = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf_aug.fit(X_train_aug, y_train_aug)
        
        # Final Prediction
        print("Generating final predictions with retrained ensemble...")
        probs_xgb_final = xgb_aug.predict_proba(X_test)
        probs_rf_final = rf_aug.predict_proba(X_test)
        
        avg_probs_final = 0.5 * probs_xgb_final + 0.5 * probs_rf_final
        preds = np.argmax(avg_probs_final, axis=1)
        
        # Update models for saving (optional, but good for inference)
        xgb = xgb_aug
        rf = rf_aug
    else:
        print("No high-confidence samples found. Skipping retraining.")
        preds = initial_preds

    pred_labels = preds
    
    # Save Model
    import joblib
    print("Saving model to model.joblib...")
    # We save a dictionary containing the models and the feature columns
    # This is important to ensure columns match during inference
    model_data = {
        'xgb': xgb,
        'rf': rf,
        'features': train_cols.tolist()
    }
    joblib.dump(model_data, 'model.joblib')
    print("Model saved.")
    
    submission = pd.DataFrame({
        'Id': test_ids,
        'change_type': pred_labels
    })
    
    submission.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")

if __name__ == "__main__":
    generate_submission()
