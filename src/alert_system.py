def generate_alerts(df):
    """Enhanced alerts with weather & seasonal insights"""
    alerts = []
    
    # Group by unique product-store combinations
    risky_groups = df[
        (df['StockoutRisk'].isin(['HIGH', 'CRITICAL'])) & 
        (df['RecommendedReorder'] > 10) & 
        (df['PredictedDemand'] > 5) & 
        (df['DaysToStockout'] < 14)
    ].groupby(['Product ID', 'Store ID']).agg({
        'Inventory Level': 'min',
        'PredictedDemand': 'mean',
        'DaysToStockout': 'min',
        'RecommendedReorder': 'sum',
        'Region': 'first',
        'Category': 'first',
        'Weather Condition': 'first'
    }).reset_index()
    
    risky_groups = risky_groups.head(8)
    
    for _, row in risky_groups.iterrows():
        product_id = str(row['Product ID']).strip()
        store_id = str(row['Store ID']).strip()
        region = row['Region']
        category = row['Category']
        weather = row['Weather Condition']
        
        realistic_reorder = min(
            int(row['RecommendedReorder']), 
            int(row['PredictedDemand'] * 7),
            200
        )
        
        # Weather-based alert enhancement
        weather_msg = ""
        if weather == 'rainy' and category == 'Groceries':
            weather_msg = " (Rain boosts grocery demand 18%)"
        elif weather == 'sunny' and category == 'Toys':
            weather_msg = " (Sunny weather boosts toy sales)"
        
        alerts.append({
            'product': product_id,
            'store': store_id,
            'region': region,
            'category': category,
            'stock': int(row['Inventory Level']),
            'demand': int(row['PredictedDemand']),
            'days': f"{row['DaysToStockout']:.1f}",
            'reorder': max(10, realistic_reorder),
            'urgency': '🚨 CRITICAL' if row['DaysToStockout'] < 5 else '⚠️ HIGH',
            'action': f"Order {max(10, realistic_reorder)} units within 24h{weather_msg}"
        })
    
    return alerts
