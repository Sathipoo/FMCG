import pandas as pd

def add_time_features(df):
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week
    return df

def encode_categoricals(df):
    df = pd.get_dummies(df, columns=['sku', 'store'], drop_first=True)
    return df