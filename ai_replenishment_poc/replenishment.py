import pandas as pd
import numpy as np

def calculate_replenishment(df, model, lead_time_days=3, safety_stock_factor=1.2):
    """
    Generate replenishment quantity using predicted demand.
    """
    df = df.copy()
    X_future = df.drop(columns=['date', 'sales_qty'])
    
    # Predict future demand
    df['predicted_demand'] = model.predict(X_future)
    
    # Calculate replenishment quantity
    df['reorder_point'] = df['predicted_demand'] * lead_time_days * safety_stock_factor
    df['current_stock'] = np.random.randint(50, 300, size=len(df))  # Simulated current stock
    df['replenishment_qty'] = df['reorder_point'] - df['current_stock']
    df['replenishment_qty'] = df['replenishment_qty'].apply(lambda x: max(0, int(x)))
    
    return df[['date', 'predicted_demand', 'reorder_point', 'current_stock', 'replenishment_qty']]