
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_data(train_path: str, test_path: str, rows: slice = None) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Loads train and test GeoJSONs.
    Args:
        rows: slice object to load a subset of rows (e.g. slice(0, 100))
    """
    print("Loading data...")
    try:
        if rows:
            train_df = gpd.read_file(train_path, rows=rows)
            test_df = gpd.read_file(test_path, rows=rows)
        else:
            train_df = gpd.read_file(train_path)
            test_df = gpd.read_file(test_path)
    except Exception as e:
        print(f"Error loading data with index_col logic, trying default: {e}")
        # Fallback without index_col if it fails or if rows is used
        train_df = gpd.read_file(train_path)
        test_df = gpd.read_file(test_path)

    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df

def engineer_features(df: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Extracts features from the GeoDataFrame.
    """
    print("Engineering features...")
    df = df.copy()
    
    # 1. Geometry Features
    df['area'] = df.geometry.area
    df['perimeter'] = df.geometry.length
    # Compactness (Polsby-Popper score or similar)
    df['compactness'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2)
    df['compactness'] = df['compactness'].fillna(0)
    
    # 2. Date Features
    # Format: DD-MM-YYYY
    # There are date0 to date4
    date_cols = [f'date{i}' for i in range(5)]
    
    for col in date_cols:
        if col in df.columns:
            # Coerce errors to NaT
            df[f'{col}_dt'] = pd.to_datetime(df[col], format='%d-%m-%Y', errors='coerce')
            # Extract basic date components (optional, maybe numeric timestamp is better)
            # df[f'{col}_timestamp'] = df[f'{col}_dt'].astype('int64') // 10**9

    # Calculate durations (days)
    for i in range(4):
        start_col = f'date{i}_dt'
        end_col = f'date{i+1}_dt'
        if start_col in df.columns and end_col in df.columns:
            df[f'duration_{i}_{i+1}'] = (df[end_col] - df[start_col]).dt.days.fillna(0)

    # 3. Categorical Features (Urban/Geography)
    # They are values like "Sparse Urban" or "Sparse Forest,Farms"
    # We treat them as Multi-Label
    
    def parse_multilabel(s):
        if pd.isna(s):
            return []
        return [x.strip() for x in s.split(',')]

    for col in ['urban_type', 'geography_type']:
        if col in df.columns:
            # Get all unique tags
            df[f'{col}_list'] = df[col].apply(parse_multilabel)
            mlb = MultiLabelBinarizer()
            encoded = mlb.fit_transform(df[f'{col}_list'])
            # Create column names
            classes = [f"{col}_{c.replace(' ', '_')}" for c in mlb.classes_]
            encoded_df = pd.DataFrame(encoded, columns=classes, index=df.index)
            df = pd.concat([df, encoded_df], axis=1)
            df = df.drop(columns=[col, f'{col}_list'])

    # 4. Change Status Features
    # change_status_date0 to date4. These are categorical.
    status_cols = [f'change_status_date{i}' for i in range(5)]
    df = pd.get_dummies(df, columns=[c for c in status_cols if c in df.columns], prefix='status')
    
    # 5. Image Statistics
    # date0 corresponds to img_..._date1
    # We can just keep them as raw features.
    # Maybe add aggregated stats (mean across all 5 dates?)
    
    img_means = [c for c in df.columns if 'mean' in c]
    img_stds = [c for c in df.columns if 'std' in c]
    
    if img_means:
        df['img_overall_mean'] = df[img_means].mean(axis=1)
        df['img_overall_std'] = df[img_stds].mean(axis=1)
    
    # Drop non-numeric for ML (dates, geometry if not needed anymore)
    # keeping geometry for visualization or spatial split if needed, but usually dropped for Sklearn
    # We will drop geometry and date objects before returning (or verification step)
    
    # Remove duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()]
    
    cols_to_drop = [c for c in df.columns if 'date' in c and '_dt' not in c and 'img' not in c and 'duration' not in c and 'status' not in c]
    # Drop datetime objects using select_dtypes
    cols_to_drop += df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    # Also drop original date strings
    df = df.drop(columns=date_cols, errors='ignore')
    # Drop datetime objects
    dt_cols = [c for c in df.columns if c.endswith('_dt')]
    df = df.drop(columns=dt_cols, errors='ignore')

    return df

if __name__ == "__main__":
    # Load sample to test
    try:
        train_df, test_df = load_data('dados/train.geojson', 'dados/test.geojson', rows=slice(0, 100))
        
        print("\nOriginal Columns:", train_df.columns.tolist()[:10])
        
        processed_train = engineer_features(train_df)
        print("\nProcessed Shape:", processed_train.shape)
        print("Processed Columns:", processed_train.columns.tolist())
        print("\nHead:\n", processed_train.head(2))
        
    except Exception as e:
        if str(e) == "'DataFrame' object has no attribute 'dtype'":
            import traceback
            traceback.print_exc()
        print(f"An error occurred: {e}")
