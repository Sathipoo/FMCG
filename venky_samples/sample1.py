import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

class InventoryReplenishmentSystem:
    """
    AI-powered inventory replenishment system that uses machine learning
    to forecast demand, optimize stock levels, and generate replenishment orders
    """
    
    def __init__(self, lead_time_days=3, service_level=0.95, holding_cost_rate=0.25, stockout_cost_multiplier=5):
        """
        Initialize the replenishment system with operational parameters
        
        Parameters:
        -----------
        lead_time_days : int
            Number of days it takes for ordered inventory to arrive
        service_level : float
            Target service level (probability of not stocking out)
        holding_cost_rate : float
            Annual holding cost as a fraction of item value
        stockout_cost_multiplier : float
            Multiplier for stockout cost relative to item price
        """
        self.lead_time_days = lead_time_days
        self.service_level = service_level
        self.holding_cost_rate = holding_cost_rate
        self.stockout_cost_multiplier = stockout_cost_multiplier
        self.models = {}
        self.scalers = {}
        self.forecast_horizon = 30  # Default forecast horizon in days
        
    def preprocess_data(self, sales_data, product_data=None, store_data=None):
        """
        Preprocess sales and product data for analysis
        
        Parameters:
        -----------
        sales_data : pandas DataFrame
            Historical sales data with columns ['date', 'product_id', 'store_id', 'quantity']
        product_data : pandas DataFrame, optional
            Product information with columns ['product_id', 'category', 'price', etc.]
        store_data : pandas DataFrame, optional
            Store information with columns ['store_id', 'location', 'size', etc.]
            
        Returns:
        --------
        pandas DataFrame
            Processed data ready for modeling
        """
        # Ensure date is in datetime format
        sales_data['date'] = pd.to_datetime(sales_data['date'])
        
        # Sort by date for time series analysis
        sales_data = sales_data.sort_values('date')
        
        # Aggregate data by day
        daily_sales = sales_data.groupby(['date', 'product_id', 'store_id'])['quantity'].sum().reset_index()
        
        # Create unique identifier for each product-store combination
        daily_sales['sku_store_id'] = daily_sales['product_id'].astype(str) + '_' + daily_sales['store_id'].astype(str)
        
        # Engineer time-based features
        daily_sales['day_of_week'] = daily_sales['date'].dt.dayofweek
        daily_sales['month'] = daily_sales['date'].dt.month
        daily_sales['year'] = daily_sales['date'].dt.year
        daily_sales['day_of_month'] = daily_sales['date'].dt.day
        daily_sales['is_weekend'] = daily_sales['day_of_week'].isin([5, 6]).astype(int)
        
        # Add rolling statistics to capture trends and seasonality
        features = []
        
        # Process each product-store combination separately
        for sku_store in daily_sales['sku_store_id'].unique():
            sku_data = daily_sales[daily_sales['sku_store_id'] == sku_store].copy()
            
            # Create lags (previous days' sales)
            for lag in [1, 3, 7, 14, 30]:
                sku_data[f'lag_{lag}d'] = sku_data['quantity'].shift(lag)
            
            # Create rolling means (moving averages)
            for window in [3, 7, 14, 30]:
                sku_data[f'rolling_mean_{window}d'] = sku_data['quantity'].rolling(window=window).mean()
                
            # Create rolling standard deviation (to capture variability)
            for window in [7, 14, 30]:
                sku_data[f'rolling_std_{window}d'] = sku_data['quantity'].rolling(window=window).std()
            
            # Add trends
            sku_data['trend_7d'] = sku_data[f'rolling_mean_7d'] - sku_data[f'rolling_mean_7d'].shift(7)
            sku_data['trend_14d'] = sku_data[f'rolling_mean_14d'] - sku_data[f'rolling_mean_14d'].shift(14)
            
            # Calculate days since last zero sale (stockout indicator)
            sku_data['zero_sale'] = (sku_data['quantity'] == 0).astype(int)
            sku_data['days_since_stockout'] = sku_data['zero_sale'].cumsum()
            sku_data['days_since_stockout'] -= sku_data['days_since_stockout'].where(sku_data['zero_sale'] == 1).fillna(method='ffill').fillna(0)
            
            features.append(sku_data)
        
        processed_data = pd.concat(features)
        
        # Merge with product and store data if provided
        if product_data is not None:
            processed_data = processed_data.merge(product_data, on='product_id', how='left')
        
        if store_data is not None:
            processed_data = processed_data.merge(store_data, on='store_id', how='left')
            
        return processed_data
    
    def add_external_features(self, data, holidays=None, promotions=None, weather=None):
        """
        Add external features that may affect demand
        
        Parameters:
        -----------
        data : pandas DataFrame
            Processed sales data
        holidays : pandas DataFrame, optional
            Holiday information with columns ['date', 'holiday_name', 'significance']
        promotions : pandas DataFrame, optional
            Promotion information with columns ['start_date', 'end_date', 'product_id', 'discount']
        weather : pandas DataFrame, optional
            Weather information with columns ['date', 'store_id', 'temperature', 'precipitation']
            
        Returns:
        --------
        pandas DataFrame
            Enhanced data with external features
        """
        enhanced_data = data.copy()
        
        # Add holiday indicators and effects
        if holidays is not None:
            holidays['date'] = pd.to_datetime(holidays['date'])
            
            # Create binary holiday indicator
            enhanced_data = enhanced_data.merge(
                holidays[['date', 'holiday_name']].assign(is_holiday=1),
                on='date', how='left'
            )
            enhanced_data['is_holiday'] = enhanced_data['is_holiday'].fillna(0)
            
            # Add days before/after holiday feature
            holiday_dates = set(holidays['date'].dt.date)
            
            enhanced_data['days_to_next_holiday'] = np.nan
            enhanced_data['days_since_last_holiday'] = np.nan
            
            for idx, row in enhanced_data.iterrows():
                current_date = row['date'].date()
                
                # Calculate days to next holiday
                days_to_next = float('inf')
                for holiday_date in holiday_dates:
                    if holiday_date > current_date:
                        days_to_next = min(days_to_next, (holiday_date - current_date).days)
                
                # Calculate days since last holiday
                days_since_last = float('inf')
                for holiday_date in holiday_dates:
                    if holiday_date < current_date:
                        days_since_last = min(days_since_last, (current_date - holiday_date).days)
                
                if days_to_next != float('inf'):
                    enhanced_data.at[idx, 'days_to_next_holiday'] = days_to_next
                    
                if days_since_last != float('inf'):
                    enhanced_data.at[idx, 'days_since_last_holiday'] = days_since_last
            
            # Fill remaining NaNs
            enhanced_data['days_to_next_holiday'] = enhanced_data['days_to_next_holiday'].fillna(30)
            enhanced_data['days_since_last_holiday'] = enhanced_data['days_since_last_holiday'].fillna(30)
        
        # Add promotion effects
        if promotions is not None:
            promotions['start_date'] = pd.to_datetime(promotions['start_date'])
            promotions['end_date'] = pd.to_datetime(promotions['end_date'])
            
            enhanced_data['on_promotion'] = 0
            enhanced_data['discount_pct'] = 0
            
            for idx, row in enhanced_data.iterrows():
                product_id = row['product_id']
                date = row['date']
                
                # Check if product is on promotion for this date
                product_promos = promotions[promotions['product_id'] == product_id]
                active_promo = product_promos[(product_promos['start_date'] <= date) & 
                                            (product_promos['end_date'] >= date)]
                
                if not active_promo.empty:
                    enhanced_data.at[idx, 'on_promotion'] = 1
                    enhanced_data.at[idx, 'discount_pct'] = active_promo['discount'].values[0]
        
        # Add weather effects
        if weather is not None:
            weather['date'] = pd.to_datetime(weather['date'])
            enhanced_data = enhanced_data.merge(
                weather, on=['date', 'store_id'], how='left'
            )
            
            # Fill missing weather data with average values
            for col in ['temperature', 'precipitation']:
                if col in enhanced_data.columns:
                    enhanced_data[col] = enhanced_data[col].fillna(enhanced_data[col].mean())
        
        return enhanced_data
    
    def train_model(self, data, sku_store_ids=None):
        """
        Train demand forecasting models for each product-store combination
        
        Parameters:
        -----------
        data : pandas DataFrame
            Preprocessed data with features
        sku_store_ids : list, optional
            List of specific product-store combinations to model. If None, all combinations are modeled.
            
        Returns:
        --------
        dict
            Dictionary of trained models keyed by sku_store_id
        """
        if sku_store_ids is None:
            sku_store_ids = data['sku_store_id'].unique()
        
        for sku_store_id in sku_store_ids:
            print(f"Training model for {sku_store_id}")
            
            # Filter data for this SKU-store combination
            sku_data = data[data['sku_store_id'] == sku_store_id].copy()
            
            # Drop rows with missing values (happens for early dates due to lag features)
            sku_data = sku_data.dropna()
            
            if len(sku_data) < 30:
                print(f"Not enough data for {sku_store_id}, skipping.")
                continue
            
            # Define features and target
            feature_cols = [col for col in sku_data.columns if col.startswith(('lag_', 'rolling_', 'trend_', 'days_since_'))]
            feature_cols += ['day_of_week', 'month', 'is_weekend', 'day_of_month']
            
            # Add external features if available
            external_features = ['is_holiday', 'days_to_next_holiday', 'days_since_last_holiday', 
                               'on_promotion', 'discount_pct', 'temperature', 'precipitation']
            
            for ef in external_features:
                if ef in sku_data.columns:
                    feature_cols.append(ef)
            
            X = sku_data[feature_cols]
            y = sku_data['quantity']
            
            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # Create a preprocessing and modeling pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestRegressor(random_state=42))
            ])
            
            # Define hyperparameter grid
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20],
                'model__min_samples_split': [2, 5, 10]
            }
            
            # Perform grid search to find best hyperparameters
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Evaluate on validation set
            y_pred = best_model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            print(f"Validation MAE: {mae:.2f}, RMSE: {rmse:.2f}")
            
            # Save the model and scaler
            self.models[sku_store_id] = best_model
            
            # Feature importance analysis
            if hasattr(best_model.named_steps['model'], 'feature_importances_'):
                importances = best_model.named_steps['model'].feature_importances_
                features = X.columns
                importance_df = pd.DataFrame({'feature': features, 'importance': importances})
                importance_df = importance_df.sort_values('importance', ascending=False)
                print("Top 5 important features:")
                print(importance_df.head(5))
            
        return self.models
    
    def forecast_demand(self, data, forecast_days=30, sku_store_ids=None):
        """
        Generate demand forecasts for the specified forecast horizon
        
        Parameters:
        -----------
        data : pandas DataFrame
            The latest available data to use as a basis for forecasting
        forecast_days : int
            Number of days to forecast into the future
        sku_store_ids : list, optional
            List of specific product-store combinations to forecast. If None, all trained models are used.
            
        Returns:
        --------
        pandas DataFrame
            Forecasted demand for each product-store combination and date
        """
        if sku_store_ids is None:
            sku_store_ids = list(self.models.keys())
        
        # Get the latest date in the data
        latest_date = data['date'].max()
        
        # Create dataframe to store forecasts
        forecasts = []
        
        for sku_store_id in sku_store_ids:
            if sku_store_id not in self.models:
                print(f"No trained model found for {sku_store_id}, skipping.")
                continue
            
            # Get the model
            model = self.models[sku_store_id]
            
            # Filter data for this SKU-store
            sku_data = data[data['sku_store_id'] == sku_store_id].copy()
            
            # Extract product_id and store_id from the sku_store_id
            product_id, store_id = sku_store_id.split('_')
            
            # Initialize forecast data
            forecast_dates = [latest_date + timedelta(days=i+1) for i in range(forecast_days)]
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'product_id': product_id,
                'store_id': store_id,
                'sku_store_id': sku_store_id
            })
            
            # Generate time features for forecast period
            forecast_df['day_of_week'] = forecast_df['date'].dt.dayofweek
            forecast_df['month'] = forecast_df['date'].dt.month
            forecast_df['year'] = forecast_df['date'].dt.year
            forecast_df['day_of_month'] = forecast_df['date'].dt.day
            forecast_df['is_weekend'] = forecast_df['day_of_week'].isin([5, 6]).astype(int)
            
            # Add external features if available in the original data
            external_features = ['is_holiday', 'days_to_next_holiday', 'days_since_last_holiday', 
                                'on_promotion', 'discount_pct', 'temperature', 'precipitation']
            
            for ef in external_features:
                if ef in data.columns:
                    # For simplicity, use the last known values for external features
                    # In a real implementation, you would ideally have future values for these features
                    last_values = sku_data[ef].iloc[-forecast_days:].values
                    if len(last_values) < forecast_days:
                        last_values = np.pad(last_values, (0, forecast_days - len(last_values)), 'edge')
                    forecast_df[ef] = last_values[:forecast_days]
            
            # Generate forecasts day by day
            for i in range(forecast_days):
                current_date = forecast_dates[i]
                
                # For the first forecast day, use actual historical data for lag features
                if i == 0:
                    # Get lag features
                    for lag in [1, 3, 7, 14, 30]:
                        lag_data = sku_data['quantity'].iloc[-lag:].values
                        forecast_df.loc[forecast_df['date'] == current_date, f'lag_{lag}d'] = lag_data[0]
                    
                    # Get rolling means
                    for window in [3, 7, 14, 30]:
                        rolling_mean = sku_data['quantity'].iloc[-window:].mean()
                        forecast_df.loc[forecast_df['date'] == current_date, f'rolling_mean_{window}d'] = rolling_mean
                    
                    # Get rolling standard deviations
                    for window in [7, 14, 30]:
                        rolling_std = sku_data['quantity'].iloc[-window:].std()
                        forecast_df.loc[forecast_df['date'] == current_date, f'rolling_std_{window}d'] = rolling_std
                    
                    # Calculate trends
                    forecast_df.loc[forecast_df['date'] == current_date, 'trend_7d'] = \
                        sku_data['rolling_mean_7d'].iloc[-1] - sku_data['rolling_mean_7d'].iloc[-8]
                    forecast_df.loc[forecast_df['date'] == current_date, 'trend_14d'] = \
                        sku_data['rolling_mean_14d'].iloc[-1] - sku_data['rolling_mean_14d'].iloc[-15]
                    
                    # Days since stockout
                    forecast_df.loc[forecast_df['date'] == current_date, 'days_since_stockout'] = \
                        sku_data['days_since_stockout'].iloc[-1] + 1
                
                # For subsequent days, use previously forecasted values
                else:
                    prev_date = forecast_dates[i-1]
                    prev_forecast = forecast_df.loc[forecast_df['date'] == prev_date, 'forecast_quantity'].values[0]
                    
                    # Update lag features
                    if i < 30:
                        forecast_df.loc[forecast_df['date'] == current_date, 'lag_1d'] = prev_forecast
                    
                    if i >= 3:
                        forecast_df.loc[forecast_df['date'] == current_date, 'lag_3d'] = \
                            forecast_df.loc[forecast_df['date'] == forecast_dates[i-3], 'forecast_quantity'].values[0]
                    
                    if i >= 7:
                        forecast_df.loc[forecast_df['date'] == current_date, 'lag_7d'] = \
                            forecast_df.loc[forecast_df['date'] == forecast_dates[i-7], 'forecast_quantity'].values[0]
                    
                    if i >= 14:
                        forecast_df.loc[forecast_df['date'] == current_date, 'lag_14d'] = \
                            forecast_df.loc[forecast_df['date'] == forecast_dates[i-14], 'forecast_quantity'].values[0]
                    
                    # Update rolling means
                    if i >= 3:
                        recent_forecasts = [forecast_df.loc[forecast_df['date'] == forecast_dates[i-j], 'forecast_quantity'].values[0] 
                                          for j in range(1, min(i+1, 3))]
                        historical_values = sku_data['quantity'].iloc[-(3-len(recent_forecasts)):].values if 3-len(recent_forecasts) > 0 else []
                        values = np.concatenate([historical_values, recent_forecasts])
                        forecast_df.loc[forecast_df['date'] == current_date, 'rolling_mean_3d'] = values.mean()
                    
                    if i >= 7:
                        recent_forecasts = [forecast_df.loc[forecast_df['date'] == forecast_dates[i-j], 'forecast_quantity'].values[0] 
                                          for j in range(1, min(i+1, 7))]
                        historical_values = sku_data['quantity'].iloc[-(7-len(recent_forecasts)):].values if 7-len(recent_forecasts) > 0 else []
                        values = np.concatenate([historical_values, recent_forecasts])
                        forecast_df.loc[forecast_df['date'] == current_date, 'rolling_mean_7d'] = values.mean()
                    
                    # Update days since stockout
                    forecast_df.loc[forecast_df['date'] == current_date, 'days_since_stockout'] = \
                        forecast_df.loc[forecast_df['date'] == prev_date, 'days_since_stockout'].values[0] + 1
                
                # Extract features for prediction
                feature_cols = [col for col in model.feature_names_in_]
                current_features = forecast_df.loc[forecast_df['date'] == current_date, feature_cols]
                
                # If we have any missing features, fill with most recent values
                for col in feature_cols:
                    if col not in forecast_df.columns or pd.isna(current_features[col].values[0]):
                        if col in sku_data.columns:
                            forecast_df.loc[forecast_df['date'] == current_date, col] = sku_data[col].iloc[-1]
                        else:
                            forecast_df.loc[forecast_df['date'] == current_date, col] = 0
                
                # Make prediction
                prediction = model.predict(forecast_df.loc[forecast_df['date'] == current_date, feature_cols])
                
                # Ensure prediction is non-negative
                prediction = max(0, prediction[0])
                
                # Add prediction to forecast dataframe
                forecast_df.loc[forecast_df['date'] == current_date, 'forecast_quantity'] = prediction
            
            # Add forecast to results
            forecasts.append(forecast_df[['date', 'product_id', 'store_id', 'sku_store_id', 'forecast_quantity']])
        
        # Combine all forecasts
        if forecasts:
            return pd.concat(forecasts)
        else:
            return pd.DataFrame(columns=['date', 'product_id', 'store_id', 'sku_store_id', 'forecast_quantity'])
    
    def calculate_safety_stock(self, demand_forecast, historical_data):
        """
        Calculate optimal safety stock levels based on service level and demand variability
        
        Parameters:
        -----------
        demand_forecast : pandas DataFrame
            Forecasted demand for each product-store combination
        historical_data : pandas DataFrame
            Historical sales data for calculating demand variability
            
        Returns:
        --------
        pandas DataFrame
            Safety stock levels for each product-store combination
        """
        from scipy.stats import norm
        
        # Get the z-score for desired service level
        z_score = norm.ppf(self.service_level)
        
        # Initialize safety stock dataframe
        safety_stock = []
        
        for sku_store_id in demand_forecast['sku_store_id'].unique():
            # Extract product_id and store_id
            product_id, store_id = sku_store_id.split('_')
            
            # Calculate demand variability from historical data
            historical_demand = historical_data[
                (historical_data['product_id'] == product_id) & 
                (historical_data['store_id'] == store_id)
            ]['quantity']
            
            # If we have sufficient historical data
            if len(historical_demand) >= 30:
                # Calculate standard deviation of demand
                demand_std = historical_demand.std()
                
                # Calculate average lead time demand
                avg_lead_time_demand = historical_demand.rolling(window=self.lead_time_days).mean().mean()
                
                # Calculate standard deviation of lead time demand
                lead_time_demand_std = historical_demand.rolling(window=self.lead_time_days).std().mean()
                
                # Calculate safety stock using formula: Z * σ_LT
                # where σ_LT is the standard deviation of demand during lead time
                ss = z_score * lead_time_demand_std
                
                # Ensure safety stock is non-negative
                ss = max(0, ss)
            else:
                # Not enough historical data, use a simplified approach
                avg_demand = historical_demand.mean() if not historical_demand.empty else 1
                ss = z_score * avg_demand * np.sqrt(self.lead_time_days)
            
            # Add to safety stock dataframe
            safety_stock.append({
                'product_id': product_id,
                'store_id': store_id,
                'sku_store_id': sku_store_id,
                'safety_stock': ss
            })
        
        return pd.DataFrame(safety_stock)
    
    def calculate_reorder_point(self, demand_forecast, safety_stock):
        """
        Calculate reorder points for each product-store combination
        
        Parameters:
        -----------
        demand_forecast : pandas DataFrame
            Forecasted demand for each product-store combination
        safety_stock : pandas DataFrame
            Calculated safety stock levels
            
        Returns:
        --------
        pandas DataFrame
            Reorder points for each product-store combination
        """
        # Initialize reorder point dataframe
        reorder_points = []
        
        for sku_store_id in demand_forecast['sku_store_id'].unique():
            # Get safety stock for this SKU-store
            ss = safety_stock.loc[safety_stock['sku_store_id'] == sku_store_id, 'safety_stock'].values[0]
            
            # Get product_id and store_id
            product_id, store_id = sku_store_id.split('_')
            
            # Calculate lead time demand from forecast
            sku_forecast = demand_forecast[demand_forecast['sku_store_id'] == sku_store_id]
            
            # Use the initial forecast days for lead time demand (or all available if less than lead time)
            max_days = min(self.lead_time_days, len(sku_forecast))
            lead_time_demand = sku_forecast['forecast_quantity'].iloc[:max_days].sum()
            
            # Calculate reorder point: Lead Time Demand + Safety Stock
            rop = lead_time_demand + ss
            
            # Add to reorder points dataframe
            reorder_points.append({
                'product_id': product_id,
                'store_id': store_id,
                'sku_store_id': sku_store_id,
                'lead_time_demand': lead_time_demand,
                'safety_stock': ss,
                'reorder_point': rop
            })
        
        return pd.DataFrame(reorder_points)
    
    def generate_order_recommendations(self, current_inventory, reorder_points, demand_forecast, economic_order_qty=None):
        """
        Generate order recommendations based on current inventory levels and reorder points
        
        Parameters:
        -----------
        current_inventory : pandas DataFrame
            Current inventory levels with columns ['product_id', 'store_id', 'current_stock']
        reorder_points : pandas DataFrame
            Calculated reorder points
        demand_forecast : pandas DataFrame
            Forecasted demand
        economic_order_qty : pandas DataFrame, optional
            Economic order quantities for each product. If None, dynamically calculated.
            
        Returns:
        --------
        pandas DataFrame
            Order recommendations with order quantities
        """
        # Create sku_store_id in current inventory
        current_inventory['sku_store_id'] = (
            current_inventory['product_id'].astype(str) + '_' + 
            current_inventory['store_id'].astype(str)
        )
        
        # Initialize order recommendations
        order_recommendations = []
        
        for sku_store_id in reorder_points['sku_store_id'].unique():
            # Get product_id and store_id
            product_id, store_id = sku_store_id.split('_')
            
            # Get current inventory level
            try:
                current_stock = current_inventory.loc[
                    current_inventory['sku_store_id'] == sku_store_id, 'current_stock'
                ].values[0]
            except (IndexError, KeyError):
                print(f"No current inventory data for {sku_store_id}, assuming zero stock.")
                current_stock = 0
            
            # Get reorder point
            rop = reorder_points.loc[
                reorder_points['sku_store_id'] == sku_store_id, 'reorder_point'
            ].values[0]
            
            # Get safety stock
            ss = reorder_points.loc[
                reorder_points['sku_store_id'] == sku_store_id, 'safety_stock'
            ].values[0]
            
            # Check if we need to reorder
            if current_stock <= rop:
                # If we have predefined EOQ
                if economic_order_qty is not None and sku_store_id in economic_order_qty['sku_store_id'].values:
                    order_qty = economic_order_qty.loc[
                        economic_order_qty['sku_store_id'] == sku_store_id, 'eoq'
                    ].values[0]
                else:
                    # Dynamically calculate order quantity based on future demand
                    sku_forecast = demand_forecast[demand_forecast['sku_store_id'] == sku_store_id]
                    
                    # Order to cover next 30 days of demand (or whatever is available in forecast)
                    future_demand = sku_forecast['forecast_quantity'].sum()
                    
                    # Order quantity = future demand + safety stock - current stock
                    order_qty = future_demand + ss - current_stock
                
 # Ensure order quantity is positive
                order_qty = max(0, order_qty)
                
                # Round up to nearest integer
                order_qty = np.ceil(order_qty)
                
                # Add to order recommendations
                order_recommendations.append({
                    'product_id': product_id,
                    'store_id': store_id,
                    'sku_store_id': sku_store_id,
                    'current_stock': current_stock,
                    'reorder_point': rop,
                    'order_quantity': order_qty,
                    'days_of_supply': order_qty / max(1, sku_forecast['forecast_quantity'].mean()) if 'sku_forecast' in locals() else None,
                    'order_date': datetime.now().strftime('%Y-%m-%d'),
                    'expected_delivery': (datetime.now() + timedelta(days=self.lead_time_days)).strftime('%Y-%m-%d')
                })
            else:
                # No order needed
                order_recommendations.append({
                    'product_id': product_id,
                    'store_id': store_id,
                    'sku_store_id': sku_store_id,
                    'current_stock': current_stock,
                    'reorder_point': rop,
                    'order_quantity': 0,
                    'days_of_supply': None,
                    'order_date': None,
                    'expected_delivery': None
                })
        
        return pd.DataFrame(order_recommendations)
    
    def calculate_economic_order_quantity(self, demand_forecast, product_data):
        """
        Calculate the economic order quantity (EOQ) for each product
        
        Parameters:
        -----------
        demand_forecast : pandas DataFrame
            Forecasted demand
        product_data : pandas DataFrame
            Product information including order cost and unit cost
            
        Returns:
        --------
        pandas DataFrame
            Economic order quantities for each product-store combination
        """
        # Initialize EOQ dataframe
        eoq_results = []
        
        for sku_store_id in demand_forecast['sku_store_id'].unique():
            # Get product_id and store_id
            product_id, store_id = sku_store_id.split('_')
            
            # Get product information
            try:
                product_info = product_data[product_data['product_id'] == product_id].iloc[0]
                unit_cost = product_info['unit_cost']
                order_cost = product_info.get('order_cost', 25)  # Default order cost if not specified
            except (IndexError, KeyError):
                print(f"No product data for {product_id}, using default values.")
                unit_cost = 10  # Default unit cost
                order_cost = 25  # Default order cost
            
            # Calculate annual demand from forecast
            sku_forecast = demand_forecast[demand_forecast['sku_store_id'] == sku_store_id]
            
            # Extrapolate annual demand from available forecast
            days_in_forecast = len(sku_forecast)
            daily_demand = sku_forecast['forecast_quantity'].mean()
            annual_demand = daily_demand * 365
            
            # Calculate holding cost per unit
            annual_holding_cost = unit_cost * self.holding_cost_rate
            
            # Calculate EOQ using the formula: sqrt(2 * D * S / H)
            # Where D = annual demand, S = order cost, H = annual holding cost per unit
            eoq = np.sqrt((2 * annual_demand * order_cost) / annual_holding_cost)
            
            # Round up to nearest integer
            eoq = np.ceil(eoq)
            
            # Add to EOQ results
            eoq_results.append({
                'product_id': product_id,
                'store_id': store_id,
                'sku_store_id': sku_store_id,
                'annual_demand': annual_demand,
                'unit_cost': unit_cost,
                'order_cost': order_cost,
                'annual_holding_cost': annual_holding_cost,
                'eoq': eoq
            })
        
        return pd.DataFrame(eoq_results)
    
    def evaluate_inventory_policy(self, order_recommendations, demand_forecast, product_data):
        """
        Evaluate the inventory policy in terms of service level, costs, and other metrics
        
        Parameters:
        -----------
        order_recommendations : pandas DataFrame
            Generated order recommendations
        demand_forecast : pandas DataFrame
            Forecasted demand
        product_data : pandas DataFrame
            Product information including costs
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        # Initialize metrics
        total_inventory_cost = 0
        total_ordering_cost = 0
        total_holding_cost = 0
        total_stockout_cost = 0
        service_level_achieved = 0
        total_demand = 0
        total_fulfilled = 0
        
        # Group forecasts by product-store
        forecast_by_sku = demand_forecast.groupby('sku_store_id')
        
        # For each product-store combination
        for sku_store_id, sku_orders in order_recommendations.groupby('sku_store_id'):
            # Get product_id
            product_id = sku_store_id.split('_')[0]
            
            # Get unit cost
            try:
                unit_cost = product_data[product_data['product_id'] == product_id]['unit_cost'].values[0]
                order_cost = product_data[product_data['product_id'] == product_id].get('order_cost', 25).values[0]
            except (IndexError, KeyError):
                print(f"No product data for {product_id}, using default values.")
                unit_cost = 10  # Default unit cost
                order_cost = 25  # Default order cost
            
            # Get forecast for this SKU
            try:
                sku_forecast = forecast_by_sku.get_group(sku_store_id)['forecast_quantity']
            except KeyError:
                print(f"No forecast for {sku_store_id}, skipping evaluation.")
                continue
            
            # Current inventory
            current_stock = sku_orders['current_stock'].values[0]
            
            # Simulate inventory over time
            inventory_levels = []
            stockout_days = 0
            
            # For each day in the forecast
            for day, demand in enumerate(sku_forecast):
                # If there's an order arriving today
                order_arrival = sku_orders[
                    pd.to_datetime(sku_orders['expected_delivery']) == 
                    pd.to_datetime(sku_forecast.index[day])
                ]
                
                if not order_arrival.empty:
                    current_stock += order_arrival['order_quantity'].sum()
                
                # Fulfill demand
                fulfilled = min(current_stock, demand)
                current_stock -= fulfilled
                
                # Record stockout if any
                if fulfilled < demand:
                    stockout_days += 1
                    stockout_quantity = demand - fulfilled
                    stockout_cost = stockout_quantity * unit_cost * self.stockout_cost_multiplier
                    total_stockout_cost += stockout_cost
                
                # Record inventory level
                inventory_levels.append(current_stock)
                
                # Add to totals
                total_demand += demand
                total_fulfilled += fulfilled
            
            # Calculate costs
            # Ordering cost: fixed cost per order
            num_orders = len(sku_orders[sku_orders['order_quantity'] > 0])
            ordering_cost = num_orders * order_cost
            total_ordering_cost += ordering_cost
            
            # Holding cost: average inventory * unit cost * holding cost rate / 365 (daily rate) * days
            avg_inventory = np.mean(inventory_levels)
            days = len(sku_forecast)
            holding_cost = avg_inventory * unit_cost * self.holding_cost_rate * days / 365
            total_holding_cost += holding_cost
            
            # Total cost for this SKU
            sku_total_cost = ordering_cost + holding_cost + total_stockout_cost
            total_inventory_cost += sku_total_cost
        
        # Calculate overall service level
        if total_demand > 0:
            service_level_achieved = total_fulfilled / total_demand
        
        # Return evaluation metrics
        evaluation = {
            'total_inventory_cost': total_inventory_cost,
            'total_ordering_cost': total_ordering_cost,
            'total_holding_cost': total_holding_cost,
            'total_stockout_cost': total_stockout_cost,
            'service_level_achieved': service_level_achieved,
            'total_demand': total_demand,
            'total_fulfilled': total_fulfilled
        }
        
        return evaluation
    
    def run_replenishment_cycle(self, sales_data, current_inventory, product_data=None, store_data=None, 
                               holidays=None, promotions=None, weather=None):
        """
        Run a complete replenishment cycle
        
        Parameters:
        -----------
        sales_data : pandas DataFrame
            Historical sales data
        current_inventory : pandas DataFrame
            Current inventory levels
        product_data : pandas DataFrame, optional
            Product information
        store_data : pandas DataFrame, optional
            Store information
        holidays : pandas DataFrame, optional
            Holiday information
        promotions : pandas DataFrame, optional
            Promotion information
        weather : pandas DataFrame, optional
            Weather information
            
        Returns:
        --------
        dict
            Dictionary containing all outputs of the replenishment cycle
        """
        print("Starting replenishment cycle...")
        
        # Step 1: Preprocess data
        print("Preprocessing data...")
        processed_data = self.preprocess_data(sales_data, product_data, store_data)
        
        # Step 2: Add external features
        print("Adding external features...")
        enhanced_data = self.add_external_features(processed_data, holidays, promotions, weather)
        
        # Step 3: Train demand forecasting models
        print("Training demand forecasting models...")
        self.train_model(enhanced_data)
        
        # Step 4: Generate demand forecasts
        print("Generating demand forecasts...")
        demand_forecast = self.forecast_demand(enhanced_data)
        
        # Step 5: Calculate safety stock
        print("Calculating safety stock...")
        safety_stock = self.calculate_safety_stock(demand_forecast, enhanced_data)
        
        # Step 6: Calculate reorder points
        print("Calculating reorder points...")
        reorder_points = self.calculate_reorder_point(demand_forecast, safety_stock)
        
        # Step
        if product_data is not None:
            # Step 7: Calculate economic order quantities
            print("Calculating economic order quantities...")
            eoq = self.calculate_economic_order_quantity(demand_forecast, product_data)
        else:
            eoq = None
        
        # Step 8: Generate order recommendations
        print("Generating order recommendations...")
        order_recommendations = self.generate_order_recommendations(
            current_inventory, reorder_points, demand_forecast, eoq
        )
        
        # Step 9: Evaluate inventory policy
        if product_data is not None:
            print("Evaluating inventory policy...")
            evaluation = self.evaluate_inventory_policy(order_recommendations, demand_forecast, product_data)
        else:
            evaluation = None
        
        print("Replenishment cycle completed.")
        
        # Return all outputs
        return {
            'processed_data': processed_data,
            'enhanced_data': enhanced_data,
            'demand_forecast': demand_forecast,
            'safety_stock': safety_stock,
            'reorder_points': reorder_points,
            'economic_order_quantity': eoq,
            'order_recommendations': order_recommendations,
            'evaluation': evaluation
        }
    
    def visualize_results(self, replenishment_results, sku_store_id=None):
        """
        Visualize the results of the replenishment cycle
        
        Parameters:
        -----------
        replenishment_results : dict
            Results from run_replenishment_cycle
        sku_store_id : str, optional
            Specific product-store combination to visualize. If None, aggregated results are shown.
        """
        # Set up the figure
        plt.figure(figsize=(15, 10))
        
        # Extract data
        demand_forecast = replenishment_results['demand_forecast']
        order_recommendations = replenishment_results['order_recommendations']
        reorder_points = replenishment_results['reorder_points']
        
        # Filter data if sku_store_id is provided
        if sku_store_id is not None:
            demand_forecast = demand_forecast[demand_forecast['sku_store_id'] == sku_store_id]
            order_recommendations = order_recommendations[order_recommendations['sku_store_id'] == sku_store_id]
            reorder_points = reorder_points[reorder_points['sku_store_id'] == sku_store_id]
            title_suffix = f" for {sku_store_id}"
        else:
            title_suffix = " (Aggregated)"
        
        # Plot 1: Demand Forecast
        plt.subplot(2, 2, 1)
        forecast_by_date = demand_forecast.groupby('date')['forecast_quantity'].sum()
        plt.plot(forecast_by_date.index, forecast_by_date.values)
        plt.title(f"Demand Forecast{title_suffix}")
        plt.xlabel("Date")
        plt.ylabel("Quantity")
        plt.grid(True)
        
        # Plot 2: Order Recommendations
        plt.subplot(2, 2, 2)
        orders = order_recommendations[order_recommendations['order_quantity'] > 0]
        plt.bar(orders['sku_store_id'], orders['order_quantity'])
        plt.title(f"Order Recommendations{title_suffix}")
        plt.xlabel("SKU-Store")
        plt.ylabel("Order Quantity")
        plt.xticks(rotation=90)
        plt.grid(True)
        
        # Plot 3: Stock Levels vs Reorder Points
        plt.subplot(2, 2, 3)
        merged = order_recommendations.merge(reorder_points, on='sku_store_id')
        plt.bar(merged['sku_store_id'], merged['current_stock'], label='Current Stock')
        plt.bar(merged['sku_store_id'], merged['reorder_point_y'], alpha=0.5, label='Reorder Point')
        plt.title(f"Stock Levels vs Reorder Points{title_suffix}")
        plt.xlabel("SKU-Store")
        plt.ylabel("Quantity")
        plt.legend()
        plt.xticks(rotation=90)
        plt.grid(True)
        
        # Plot 4: Evaluation Metrics
        if replenishment_results['evaluation'] is not None:
            plt.subplot(2, 2, 4)
            eval_metrics = replenishment_results['evaluation']
            metrics = ['total_ordering_cost', 'total_holding_cost', 'total_stockout_cost']
            values = [eval_metrics[m] for m in metrics]
            plt.bar(metrics, values)
            plt.title("Cost Breakdown")
            plt.xlabel("Cost Type")
            plt.ylabel("Cost ($)")
            plt.grid(True)
            
            # Add service level as text
            service_level = eval_metrics['service_level_achieved'] * 100
            plt.text(0.5, 0.9, f"Service Level: {service_level:.1f}%", 
                    ha='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.show()
    
    def save_models(self, filepath):
        """
        Save trained models to disk
        
        Parameters:
        -----------
        filepath : str
            Path to save the models
        """
        joblib.dump(self.models, filepath)
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath):
        """
        Load trained models from disk
        
        Parameters:
        -----------
        filepath : str
            Path to load the models from
        """
        self.models = joblib.load(filepath)
        print(f"Models loaded from {filepath}")
        return self.models


# Example Usage
if __name__ == "__main__":
    # Generate synthetic data for demonstration
    def generate_synthetic_data(n_products=5, n_stores=2, start_date='2024-01-01', end_date='2025-04-01'):
        """Generate synthetic sales data for demonstration"""
        # Date range
        date_range = pd.date_range(start=start_date, end=end_date)
        
        # Create product data
        products = []
        for p_id in range(1, n_products+1):
            products.append({
                'product_id': f'P{p_id}',
                'category': f'Category{p_id % 3 + 1}',
                'unit_cost': np.random.uniform(5, 50),
                'price': np.random.uniform(10, 100),
                'order_cost': np.random.uniform(10, 30)
            })
        product_data = pd.DataFrame(products)
        
        # Create store data
        stores = []
        for s_id in range(1, n_stores+1):
            stores.append({
                'store_id': f'S{s_id}',
                'location': f'Location{s_id}',
                'size': np.random.choice(['Small', 'Medium', 'Large']),
                'region': f'Region{s_id % 2 + 1}'
            })
        store_data = pd.DataFrame(stores)
        
        # Create sales data
        sales = []
        for date in date_range:
            for p_id in range(1, n_products+1):
                for s_id in range(1, n_stores+1):
                    # Base demand
                    base_demand = np.random.normal(50, 10)
                    
                    # Add seasonality
                    day_of_week = date.dayofweek
                    month = date.month
                    
                    # Weekend effect
                    if day_of_week >= 5:  # Weekend
                        base_demand *= 1.5
                    
                    # Monthly seasonality
                    if month in [11, 12]:  # Holiday season
                        base_demand *= 1.3
                    elif month in [1, 2]:  # Post-holiday slump
                        base_demand *= 0.8
                    
                    # Product-specific patterns
                    if p_id % 3 == 0:  # Trending product
                        days_since_start = (date - pd.to_datetime(start_date)).days
                        trend_factor = 1 + (days_since_start / 365)
                        base_demand *= trend_factor
                    
                    # Store-specific patterns
                    if s_id % 2 == 0:  # Larger stores
                        base_demand *= 1.2
                    
                    # Random noise
                    final_demand = max(0, int(base_demand * np.random.normal(1, 0.2)))
                    
                    # Add to sales data
                    sales.append({
                        'date': date,
                        'product_id': f'P{p_id}',
                        'store_id': f'S{s_id}',
                        'quantity': final_demand
                    })
        
        sales_data = pd.DataFrame(sales)
        
        # Create current inventory data
        inventory = []
        for p_id in range(1, n_products+1):
            for s_id in range(1, n_stores+1):
                inventory.append({
                    'product_id': f'P{p_id}',
                    'store_id': f'S{s_id}',
                    'current_stock': np.random.randint(20, 100)
                })
        
        inventory_data = pd.DataFrame(inventory)
        
        return sales_data, product_data, store_data, inventory_data
    
    # Generate synthetic data
    print("Generating synthetic data...")
    sales_data, product_data, store_data, inventory_data = generate_synthetic_data()
    
    # Create replenishment system
    print("Creating replenishment system...")
    replenishment_system = InventoryReplenishmentSystem(
        lead_time_days=3,
        service_level=0.95,
        holding_cost_rate=0.25
    )
    
    # Run replenishment cycle
    print("Running replenishment cycle...")
    results = replenishment_system.run_replenishment_cycle(
        sales_data=sales_data,
        current_inventory=inventory_data,
        product_data=product_data,
        store_data=store_data
    )
    
    # Visualize results
    print("Visualizing results...")
    replenishment_system.visualize_results(results)
    
    # Display order recommendations
    print("\nOrder Recommendations:")
    print(results['order_recommendations'][['product_id', 'store_id', 'current_stock', 'reorder_point', 'order_quantity']])
    
    # Display evaluation metrics
    print("\nInventory Policy Evaluation:")
    for metric, value in results['evaluation'].items():
        print(f"{metric}: {value}")
    
    # Save trained models
    replenishment_system.save_models('replenishment_models.pkl')
    
    print("Demonstration completed.")