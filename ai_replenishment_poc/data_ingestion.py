import pandas as pd

def load_sample_data(path='data/sample_sales.csv'):
    """
    Load sample sales data for demand forecasting.
    Columns expected: ['date', 'sku', 'store', 'sales_qty', 'promotion']
    """
    df = pd.read_csv(path, parse_dates=['date'])
    return df