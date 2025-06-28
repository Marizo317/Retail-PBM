# Retail_PBM - Comprehensive Dashboard & Predictive Model
# Steps 1-4: Loading, Analysis, Feature Engineering, and Model Training

# --- 1. IMPORT LIBRARIES ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def load_and_prepare_data(filename='sales_data.csv'):
    # (This function remains the same as before)
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
    # (This function remains the same as before)
    print("\n\n--- Running Profitability Analysis ---")
    df_time_analysis = df.set_index('Date')
    monthly_net_profit = df_time_analysis['Profit'].sum() - df_time_analysis['Waste_Cost'].sum()
    print("\n----------------- MONTHLY SUMMARY -----------------")
    print(f"ESTIMATED MONTHLY NET PROFIT: €{monthly_net_profit:.2f}")
    print("---------------------------------------------------")

def run_feature_engineering(df):
    # (This function remains the same as before)
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
    print("[SUCCESS] Feature Engineering complete. Data is ready for the model.")
    return df_model

# --- NEW FUNCTION FOR STEP 4 ---
def run_model_training(df_model):
    """Trains a sales prediction model and evaluates its performance."""
    print("\n\n--- Running Step 4: Model Building & Training ---")

    # --- 4.1. Define Features (X) and Target (y) ---
    # The target is what we want to predict.
    y = df_model['Quantity_Sold_kg']
    # The features are all the other columns that we use as clues.
    X = df_model.drop('Quantity_Sold_kg', axis=1)

    # --- 4.2. Split Data into Training and Testing Sets ---
    # We'll use 80% of the data to train the model and 20% to test it.
    # `random_state` ensures that the split is the same every time we run it.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

    # --- 4.3. Train the Random Forest Model ---
    # We initialize the model with some standard parameters.
    # `n_estimators` is the number of "decision trees" the model builds.
    model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
    
    print("Training the model... (This may take a moment)")
    model.fit(X_train, y_train)
    print("[SUCCESS] Model training complete.")

    # --- 4.4. Evaluate the Model ---
    # We use the test set (the "exam") to see how well the model performs.
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    oob = model.oob_score_

    print("\n--- Model Performance Evaluation ---")
    print(f"R-squared (R²): {r2:.2f}")
    print(f"Out-of-Bag (OOB) Score: {oob:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} kg")
    print("------------------------------------")
    print(f"(Interpretation: R² shows how much of the sales variation our model explains. OOB is a self-check score. MAE means our predictions are, on average, off by ±{mae:.2f} kg.)")

    # --- 4.5. Feature Importance ---
    # Let's see which clues the model found most important.
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n--- Most Important Features for the Model ---")
    print(feature_importances.head(5))
    
    return model

# ==================================================================
#                       MAIN EXECUTION BLOCK
# ==================================================================
if __name__ == "__main__":
    print("======================================================")
    print("=== Retail_PBM | Dashboard & Prediction Pipeline  ===")
    print("======================================================")
    
    main_df = load_and_prepare_data()
    
    if main_df is not None:
        run_profitability_analysis(main_df.copy())
        
        model_ready_df = run_feature_engineering(main_df.copy())
        
        # New step added here!
        trained_model = run_model_training(model_ready_df)
        
        # The visualization part (optional, can be commented out if not needed)
        print("\n\n--- Generating Visualizations from Analysis ---")
        performance_by_variety = main_df.groupby('Variety')['Profit'].sum().sort_values(ascending=False)
        plt.figure(figsize=(12, 7))
        sns.barplot(x=performance_by_variety.index, y=performance_by_variety.values, palette='viridis')
        plt.title('Total Profit per Tomato Variety (€)', fontsize=16)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        print("\n[PIPELINE COMPLETE]")