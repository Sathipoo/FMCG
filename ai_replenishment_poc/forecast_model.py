import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

def train_forecast_model(df):
    df = df.copy()
    X = df.drop(columns=['date', 'sales_qty'])
    y = df['sales_qty']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"MAPE: {mape:.2f}")
    return model