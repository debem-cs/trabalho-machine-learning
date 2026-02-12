
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
    
    # Primary metric for optimization: Weighted F1 (matches Kaggle 0.90)
    primary_metric = 'f1_weighted' 
    
    # Multiple metrics for reporting
    scorers = {
        'f1_macro': make_scorer(f1_score, average='macro'),
        'f1_weighted': make_scorer(f1_score, average='weighted'),
        'f1_micro': make_scorer(f1_score, average='micro'), # Equivalent to accuracy
        'accuracy': 'accuracy'
    }
    
    results = {}
    
    if args.tune:
        from sklearn.model_selection import RandomizedSearchCV
        print(f"\nRunning Hyperparameter Tuning (XGBoost) optimizing {primary_metric}...")
        param_dist = {
            'n_estimators': [200, 300, 400, 500], # Increased estimators
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [5, 7, 10, 12],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
            'min_child_weight': [1, 3, 5]
        }
        xgb = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss')
        search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=15, cv=3, scoring=primary_metric, verbose=1, random_state=42, n_jobs=-1)
        search.fit(X, y)
        print(f"Best XGB Params: {search.best_params_}")
        print(f"Best XGB {primary_metric}: {search.best_score_:.4f}")
        results['XGBoost_Tuned'] = search.best_score_
    else:
        # Models to test
        # Note: Removing class_weight='balanced' might actually improve Weighted F1 
        # because 'balanced' sacrifices majority class accuracy for minority recall.
        models = {
            'RandomForest_Balanced': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
            'RandomForest_Standard': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), # Test standard RF
            'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=10, random_state=42, n_jobs=-1) # Current Best
        }
        
        print("\nStarting Cross-Validation...")
        for name, model in models.items():
            print(f"Evaluating {name}...")
            # Calculate metrics independently
            for metric_name, scorer in scorers.items():
                scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
                print(f"  {metric_name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
                if metric_name == primary_metric:
                    results[name] = scores
            
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
