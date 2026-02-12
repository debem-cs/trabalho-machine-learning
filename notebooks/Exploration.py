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
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from src.features import load_data, engineer_features

# %%
# Load Data
train_df, test_df = load_data('../dados/train.geojson', '../dados/test.geojson')

# %%
# Class Distribution
plt.figure(figsize=(10, 6))
sns.countplot(y='change_type', data=train_df, order=train_df['change_type'].value_counts().index)
plt.title('Distribution of Change Types')
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

plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Feature Correlation Matrix')
plt.show()

# %%
# Plot some geometries
train_df.head(5).plot(column='change_type', legend=True, figsize=(10, 10))
plt.title('Sample Geometries')
plt.show()
