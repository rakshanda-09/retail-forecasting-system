import pandas as pd
import numpy as np

class InventoryOptimizer:
    def analyze_risk(self, df, predictions):
        df['PredictedDemand'] = predictions
        
        # Calculate standard deviation of demand to find "Volatility"
        volatility = df.groupby('Product ID')['Units Sold'].transform('std').fillna(0)
        
        # Days until empty
        df['DaysToStockout'] = np.where(
            df['PredictedDemand'] > 0, 
            df['Inventory Level'] / (df['PredictedDemand'] + 0.1), 
            999
        )
        
        # Refined Risk levels (REALISTIC thresholds)
        conditions = [
            (df['DaysToStockout'] < 5),   # Critical: <5 days
            (df['DaysToStockout'] < 14),  # High: <2 weeks
            (df['DaysToStockout'] < 30)   # Medium: <1 month
        ]
        choices = ['CRITICAL', 'HIGH', 'MEDIUM']
        df['StockoutRisk'] = np.select(conditions, choices, default='LOW')
        
        return df
    
    def generate_recommendations(self, df):
        # DYNAMIC REORDER POINT: (Demand * LeadTime) + Buffer
        lead_time = 3  # 3 days realistic lead time
        df['SafetyStock'] = (df['PredictedDemand'] * lead_time * 0.5).clip(lower=10)
        
        df['RecommendedReorder'] = np.where(
            df['Inventory Level'] < df['SafetyStock'],
            (df['PredictedDemand'] * 7) - df['Inventory Level'], # Order 1 week's worth
            0
        )
        
        # Clip reorder to reasonable numbers
        df['RecommendedReorder'] = df['RecommendedReorder'].clip(lower=0, upper=200).round(0)
        return df
