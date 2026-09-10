import pandas as pd
import numpy as np

class DataManager:
    def __init__(self):
        pass
 
    def load_data(self, file_path):
        return pd.read_csv(file_path)
 
    def clean_data(self, df):
        print("🧹 Performing Advanced Cleaning...")
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values(['Store ID', 'Product ID', 'Date']).reset_index(drop=True)
        
        # Advanced outlier handling
        for col in ['Units Sold', 'Price']:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            upper = q3 + 1.5 * iqr
            df[col] = df[col].clip(upper=upper)
        
        # Smart imputation by product-store combo
        df['Units Sold'] = df.groupby(['Product ID', 'Store ID'])['Units Sold'].transform(
            lambda x: x.fillna(x.median()).fillna(x.mean())
        )
        df['Units Sold'] = df['Units Sold'].clip(lower=0)
        df['Price'] = df['Price'].clip(lower=0.01)
        
        return df
 

    def feature_engineering(self, df):
        print("🔧 Engineering High-Correlation Features...")
        # Enhanced time features (NO HOLIDAY LOGIC)
        df['Month'] = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Quarter'] = df['Date'].dt.quarter
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Weather encoding
        weather_map = {'cloudy': 0, 'rainy': 1, 'sunny': 2, 'snowy': 3}
        df['Weather_Code'] = df['Weather Condition'].map(weather_map).fillna(0)
        
        # Season encoding
        season_map = {'winter': 0, 'spring': 1, 'summer': 2, 'autumn': 3}
        df['Season_Code'] = df['Seasonality'].map(season_map).fillna(0)
        
        # Business features
        df['FinalPrice'] = df['Price'] * (1 - df['Discount'].clip(0, 1))
        df['Revenue'] = df['Units Sold'] * df['FinalPrice']
        
        cat_avg = df.groupby('Category')['Price'].transform('mean')
        df['Price_Relative'] = df['Price'] / (cat_avg + 0.1)
        df['Inventory_Days'] = df['Inventory Level'] / (df['Units Sold'] + 1)
        
        return df.fillna(0)
