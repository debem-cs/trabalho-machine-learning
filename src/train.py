
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, f1_score
import sys
import os
# Add project root to path
sys.path.append(os.getcwd())
from src.features import load_data, engineer_features
import argparse

# Target mapping as per instructions/skeleton code
CHANGE_TYPE_MAP = {
    'Demolition': 0, 
    'Road': 1, 
    'Residential': 2, 
    'Commercial': 3, 
    'Industrial': 4,
    'Mega Projects': 5
}

def get_X_y(df):
    """
    Splits DataFrame into X and y.
    """
    if 'change_type' in df.columns:
        y = df['change_type'].map(CHANGE_TYPE_MAP)
        X = df.drop(columns=['change_type', 'index'], errors='ignore')
    else:
        y = None
        X = df.drop(columns=['index'], errors='ignore')
    
    # Ensure all columns are numeric
    X = X.select_dtypes(include=[np.number])
    return X, y

def train_and_evaluate():
    """
    Trains models and evaluating them using Cross-Validation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=None, help='Number of rows to sample for quick testing')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
    args = parser.parse_args()

    rows = slice(0, args.sample) if args.sample else None
    
    print(f"Loading and preparing data ({'sample' if rows else 'full'})...")
    train_geo, _ = load_data('dados/train.geojson', 'dados/test.geojson', rows=rows)
    train_df = engineer_features(train_geo)
    
    X, y = get_X_y(train_df)
    
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    
    # CV Strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score, average='macro')
    
    results = {}
    
    if args.tune:
        from sklearn.model_selection import RandomizedSearchCV
        print("\nRunning Hyperparameter Tuning (XGBoost)...")
        param_dist = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
        xgb = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss')
        search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=10, cv=3, scoring=scorer, verbose=1, random_state=42, n_jobs=-1)
        search.fit(X, y)
        print(f"Best XGB Params: {search.best_params_}")
        print(f"Best XGB CV Score: {search.best_score_:.4f}")
        results['XGBoost_Tuned'] = search.best_score_
    else:
        # Models to test
        models = {
            'RandomForest_Balanced': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
            'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
        }
        
        print("\nStarting Cross-Validation...")
        for name, model in models.items():
            print(f"Evaluating {name}...")
            scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
            results[name] = scores
            print(f"{name} F1-Macro: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
            # Feature Importance (on full sample)
            if hasattr(model, 'feature_importances_'):
                model.fit(X, y)
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                print(f"\nTop 20 Features ({name}):")
                for f in range(20):
                    print(f"{X.columns[indices[f]]}: {importances[indices[f]]:.4f}")
            
    return results

if __name__ == "__main__":
    train_and_evaluate()
