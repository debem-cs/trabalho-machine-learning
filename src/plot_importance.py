import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load model
model_data = joblib.load('model.joblib')
xgb_model = model_data['xgb']
feature_names = model_data['features']

# Get feature importance
importance = xgb_model.feature_importances_
feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importance})
feat_imp = feat_imp.sort_values('importance', ascending=False).head(15)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feat_imp, palette='viridis')
plt.title('Feature Weights Optimized for Maximizing Accuracy')
plt.xlabel('Relative Contribution (Gain)')
plt.ylabel('Feature')
plt.tight_layout()

# Save to Latex folder
os.makedirs('Latex', exist_ok=True)
plt.savefig('Latex/feature_importance.png')
print("Feature importance plot saved to Latex/feature_importance.png")
