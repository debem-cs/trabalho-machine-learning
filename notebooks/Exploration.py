# %% [markdown]
# # Data Exploration
# This notebook explores the dataset for the Change Detection project.

# %%
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import pandas as pd
import numpy as np

# Add src to path
# Robustly find project root
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.features import load_data, engineer_features

# %%
# Load Data
data_dir = os.path.join(project_root, 'dados')
train_path = os.path.join(data_dir, 'train.geojson')
test_path = os.path.join(data_dir, 'test.geojson')
print(f"Loading data from: {train_path}")
train_df, test_df = load_data(train_path, test_path)

# %%
# Setup Logs Directory
logs_dir = os.path.join(project_root, 'logs')
os.makedirs(logs_dir, exist_ok=True)
print(f"Saving plots to: {logs_dir}")

# %%
# Class Distribution
plt.figure(figsize=(12, 8))
sns.countplot(y='change_type', data=train_df, order=train_df['change_type'].value_counts().index)
plt.title('Distribution of Change Types')
plt.tight_layout()
plt.savefig(os.path.join(logs_dir, 'class_distribution.png'))
plt.show()

# %% [markdown]
# ## Feature Engineering
# Let's apply our feature engineering pipeline and inspect the result.

# %%
train_processed = engineer_features(train_df)
print("Processed shape:", train_processed.shape)

# %%
# Correlation Matrix of Numerical Features
numeric_cols = train_processed.select_dtypes(include=[np.number]).columns
corr = train_processed[numeric_cols].corr()

plt.figure(figsize=(20, 18)) # Increased size to prevent cutting off
sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(logs_dir, 'correlation_matrix.png'))
plt.show()

# %%
# Plot some geometries
train_df.head(5).plot(column='change_type', legend=True, figsize=(12, 12))
plt.title('Sample Geometries')
plt.tight_layout()
plt.savefig(os.path.join(logs_dir, 'sample_geometries.png'))
plt.show()
