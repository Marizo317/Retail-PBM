# Retail_PBM - Comprehensive Dashboard & Predictive Model
# Final Version: Includes Analysis, Training, and Future Prediction

# --- 1. IMPORT LIBRARIES ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def load_and_prepare_data(filename='sales_data.csv'):
    """Loads, normalizes, and calculates initial metrics for the dataset."""
    print("\n--- Running Step 1 & 2: Data Loading and Preparation ---")
    spanish_to_english_map = {
        'Fecha': 'Date', 'Variedad': 'Variety', 'Cantidad_Vendida_kg': 'Quantity_Sold_kg',
        'Stock_Inicial_kg': 'Initial_Stock_kg', 'Precio_por_kg': 'Price_per_kg', 'Coste_por_kg': 'Cost_per_kg'
    }
    try:
        df = pd.read_csv(filename)
        df.rename(columns=spanish_to_english_map, inplace=True)
        print(f"[SUCCESS] File '{filename}' loaded.")
    except FileNotFoundError:
        print(f"[ERROR] The file '{filename}' was not found.")
        return None
    df['Date'] = pd.to_datetime(df['Date'])
    df['Revenue'] = df['Quantity_Sold_kg'] * df['Price_per_kg']
    df['Profit'] = (df['Price_per_kg'] - df['Cost_per_kg']) * df['Quantity_Sold_kg']
    df['Potential_Waste_kg'] = (df['Initial_Stock_kg'] - df['Quantity_Sold_kg']).clip(lower=0)
    df['Waste_Cost'] = df['Potential_Waste_kg'] * df['Cost_per_kg']
    print("[SUCCESS] Core business metrics calculated.")
    return df

def run_profitability_analysis(df):
    """Runs and prints the time-based and product-based profitability analysis."""
    print("\n\n--- Running Comprehensive Analysis Dashboard ---")
    
    # --- Part 1: Time-Based Financial Report ---
    df_time_analysis = df.set_index('Date')
    
    # -- Monthly Analysis --
    monthly_profit = df_time_analysis['Profit'].sum()
    monthly_waste_cost = df_time_analysis['Waste_Cost'].sum()
    monthly_net_profit = monthly_profit - monthly_waste_cost
    monthly_profit_status = " PROFITABLE" if monthly_net_profit > 0 else " NOT PROFITABLE"
    
    print("\n----------------- MONTHLY SUMMARY -----------------")
    print(f"Estimated Monthly Net Profit: €{monthly_net_profit:.2f}")
    print(f"Monthly Status: {monthly_profit_status}")
    print("---------------------------------------------------")

    # -- Weekly Analysis --
    weekly_summary = df_time_analysis.resample('W-Mon').agg(
        Weekly_Gross_Profit=('Profit', 'sum'),
        Weekly_Waste_Cost=('Waste_Cost', 'sum')
    )
    weekly_summary['Weekly_Net_Profit'] = weekly_summary['Weekly_Gross_Profit'] - weekly_summary['Weekly_Waste_Cost']
    weekly_summary['Is_Profitable'] = weekly_summary['Weekly_Net_Profit'].apply(lambda x: "✅ Yes" if x > 0 else "❌ No")
    
    print("\n\n----------------- WEEKLY SUMMARY -----------------")
    weekly_summary.index = weekly_summary.index.strftime('Week of %Y-%m-%d')
    print(weekly_summary[['Weekly_Net_Profit', 'Is_Profitable']])
    print("--------------------------------------------------")

    # --- Part 2: Product-Based Performance Report ---
    print("\n\n--- PRODUCT PERFORMANCE REPORT ---")
    
    performance_by_variety = df.groupby('Variety').agg(
        Total_Quantity_Sold=('Quantity_Sold_kg', 'sum'),
        Total_Gross_Profit=('Profit', 'sum'),
        Total_Waste_Cost=('Waste_Cost', 'sum')
    ).sort_values(by='Total_Quantity_Sold', ascending=False)
    
    performance_by_variety['Total_Net_Profit'] = performance_by_variety['Total_Gross_Profit'] - performance_by_variety['Total_Waste_Cost']

    print("\nFull Performance Table by Variety:")
    print(performance_by_variety)

    top_3_best_sellers_by_qty = performance_by_variety['Total_Quantity_Sold'].head(3)
    top_3_worst_sellers_by_qty = performance_by_variety['Total_Quantity_Sold'].tail(3)
    top_3_most_profitable = performance_by_variety.sort_values(by='Total_Net_Profit', ascending=False)['Total_Net_Profit'].head(3)

    print("\n\n Top 3 Best-Sellers (by Quantity):")
    print(top_3_best_sellers_by_qty)

    print("\n Top 3 Worst-Sellers (by Quantity):")
    print(top_3_worst_sellers_by_qty)

    print("\n Top 3 Most Profitable (by Net Profit):")
    print(top_3_most_profitable)

def run_feature_engineering(df):
    """Prepares the data for machine learning by creating numerical features."""
    print("\n\n--- Running Step 3: Data Preparation for Prediction ---")
    df_model = df[['Date', 'Variety', 'Quantity_Sold_kg']].copy()
    df_model['day_of_week'] = df_model['Date'].dt.dayofweek
    df_model['day_of_month'] = df_model['Date'].dt.day
    df_model['week_of_year'] = df_model['Date'].dt.isocalendar().week
    df_model['month'] = df_model['Date'].dt.month
    df_model = df_model.sort_values(by=['Variety', 'Date'])
    df_model['sales_lag_1_day'] = df_model.groupby('Variety')['Quantity_Sold_kg'].shift(1)
    df_model = pd.get_dummies(df_model, columns=['Variety'], prefix='Variety')
    df_model.dropna(inplace=True)
    df_model = df_model.drop('Date', axis=1)
    print("[SUCCESS] Feature Engineering complete.")
    return df_model

def run_model_training(df_model):
    """Trains a sales prediction model and evaluates its performance."""
    print("\n\n--- Running Step 4: Model Building & Training ---")
    y = df_model['Quantity_Sold_kg']
    X = df_model.drop('Quantity_Sold_kg', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("[SUCCESS] Model training complete.")

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print("\n--- Model Performance Evaluation ---")
    print(f"R-squared (R²): {r2:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} kg")
    print("------------------------------------")
    
    return model, X.columns

def generate_future_predictions(model, df_full, model_columns):
    """Uses the trained model to predict sales for the next day and provides recommendations."""
    print("\n\n--- Running Step 5: Prediction & Recommendation ---")
    
    last_date_in_data = df_full['Date'].max()
    prediction_date = last_date_in_data + pd.Timedelta(days=1)
    
    print(f"\nGenerating purchase recommendations for: {prediction_date.strftime('%Y-%m-%d')}")
    print("-------------------------------------------------")
    
    varieties = df_full['Variety'].unique()
    predictions_data = []

    for variety in varieties:
        latest_sale_kg = df_full[df_full['Variety'] == variety].sort_values(by='Date').iloc[-1]['Quantity_Sold_kg']
        
        features = {
            'day_of_week': prediction_date.dayofweek, 'day_of_month': prediction_date.day,
            'week_of_year': prediction_date.isocalendar().week, 'month': prediction_date.month,
            'sales_lag_1_day': latest_sale_kg
        }
        
        for col in model_columns:
            if col.startswith('Variety_') and col != f'Variety_{variety}':
                features[col] = 0
        features[f'Variety_{variety}'] = 1

        prediction_df = pd.DataFrame([features], columns=model_columns)
        predicted_kg = model.predict(prediction_df)
        
        recommendation = f"Stock at least {predicted_kg[0] + 5:.2f} kg"
        predictions_data.append({
            'Variety': variety,
            'Predicted_Sales_kg': f"{predicted_kg[0]:.2f}",
            'Recommendation': recommendation
        })
    
    recommendations_df = pd.DataFrame(predictions_data)
    print(recommendations_df.to_string(index=False))
    print("-------------------------------------------------")

# ==================================================================
#                       MAIN EXECUTION BLOCK
# ==================================================================
if __name__ == "__main__":
    main_df = load_and_prepare_data()
    
    if main_df is not None:
        run_profitability_analysis(main_df.copy())
        model_ready_df = run_feature_engineering(main_df.copy())
        trained_model, model_columns = run_model_training(model_ready_df)
        generate_future_predictions(trained_model, main_df.copy(), model_columns)
        
        print("\n\n--- Generating Visualizations from Analysis ---")
        performance_by_variety = main_df.groupby('Variety')['Profit'].sum().sort_values(ascending=False)
        plt.figure(figsize=(12, 7))
        sns.barplot(x=performance_by_variety.index, y=performance_by_variety.values, palette='viridis')
        plt.title('Total Profit per Tomato Variety (€)', fontsize=16)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        print("\n[PROJECT EXECUTION COMPLETE]")