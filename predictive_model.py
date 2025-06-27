# Retail_PBM - Comprehensive Profitability & Performance Dashboard
# This script combines time-based profitability analysis with product-based performance analysis.

# --- 1. IMPORT LIBRARIES ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_comprehensive_analysis():
    """
    Main function to run the entire analysis pipeline.
    """
    print("======================================================")
    print("=== Retail_PBM | Comprehensive Analysis Dashboard ===")
    print("======================================================")

    # --- 2. DATA LOADING AND PREPARATION ---
    filename = 'sales_data.csv'
    spanish_to_english_map = {
        'Fecha': 'Date', 'Variedad': 'Variety',
        'Cantidad_Vendida_kg': 'Quantity_Sold_kg',
        'Stock_Inicial_kg': 'Initial_Stock_kg',
        'Precio_por_kg': 'Price_per_kg', 'Coste_por_kg': 'Cost_per_kg'
    }

    try:
        df = pd.read_csv(filename)
        df.rename(columns=spanish_to_english_map, inplace=True)
        print(f"\n[SUCCESS] File '{filename}' loaded and columns normalized.")
    except FileNotFoundError:
        print(f"\n[ERROR] The file '{filename}' was not found.")
        return # Stop execution

    # --- 3. CORE CALCULATIONS ---
    # Convert 'Date' column to datetime objects
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate financial and stock metrics for each row
    df['Revenue'] = df['Quantity_Sold_kg'] * df['Price_per_kg']
    df['Profit'] = (df['Price_per_kg'] - df['Cost_per_kg']) * df['Quantity_Sold_kg']
    df['Potential_Waste_kg'] = df['Initial_Stock_kg'] - df['Quantity_Sold_kg']
    df['Potential_Waste_kg'] = df['Potential_Waste_kg'].clip(lower=0)
    df['Waste_Cost'] = df['Potential_Waste_kg'] * df['Cost_per_kg']

    print("[SUCCESS] Core business metrics calculated (Profit, Waste, etc).")

    # --- 4. TIME-BASED PROFITABILITY REPORT ---
    print("\n\n\n--- REPORT 1: Financial Performance Over Time ---")
    
    # Create a temporary dataframe with Date as index for time analysis
    df_time_analysis = df.set_index('Date')
    
    # Monthly Analysis
    monthly_profit = df_time_analysis['Profit'].sum()
    monthly_waste_cost = df_time_analysis['Waste_Cost'].sum()
    monthly_net_profit = monthly_profit - monthly_waste_cost
    
    print("\n----------------- MONTHLY SUMMARY -----------------")
    print(f"Gross Profit (from sales): €{monthly_profit:.2f}")
    print(f"Estimated Cost of Waste:   €{monthly_waste_cost:.2f}")
    print("---------------------------------------------------")
    print(f"ESTIMATED MONTHLY NET PROFIT: €{monthly_net_profit:.2f}")
    print("---------------------------------------------------")

    # Weekly Analysis
    weekly_summary = df_time_analysis.resample('W-Mon').agg(
        Weekly_Gross_Profit=('Profit', 'sum'),
        Weekly_Waste_Cost=('Waste_Cost', 'sum')
    )
    weekly_summary['Weekly_Net_Profit'] = weekly_summary['Weekly_Gross_Profit'] - weekly_summary['Weekly_Waste_Cost']
    
    print("\n\n----------------- WEEKLY SUMMARY -----------------")
    weekly_summary.index = weekly_summary.index.strftime('Week of %Y-%m-%d')
    print(weekly_summary)
    print("--------------------------------------------------")

    # --- 5. PRODUCT-BASED PERFORMANCE REPORT ---
    print("\n\n\n--- REPORT 2: Performance by Product Variety ---")
    
    performance_by_variety = df.groupby('Variety').agg(
        Total_Quantity_Sold=('Quantity_Sold_kg', 'sum'),
        Total_Gross_Profit=('Profit', 'sum'),
        Total_Waste_Cost=('Waste_Cost', 'sum')
    ).sort_values(by='Total_Quantity_Sold', ascending=False)
    
    performance_by_variety['Total_Net_Profit'] = performance_by_variety['Total_Gross_Profit'] - performance_by_variety['Total_Waste_Cost']

    print("\nFull Performance Table by Variety:")
    print(performance_by_variety)

    # Identify Top 3 and Bottom 3 performers
    print("\n\n🏆 Top 3 Best-Sellers (by Quantity):")
    print(performance_by_variety['Total_Quantity_Sold'].head(3))

    print("\n📉 Top 3 Worst-Sellers (by Quantity):")
    print(performance_by_variety['Total_Quantity_Sold'].tail(3))
    
    print("\n💰 Top 3 Most Profitable (by Net Profit):")
    print(performance_by_variety.sort_values(by='Total_Net_Profit', ascending=False)['Total_Net_Profit'].head(3))
    
    # --- 6. VISUALIZATION ---
    print("\n\n--- Generating Visualizations ---")
    sns.set_style("whitegrid")

    # Plot 1: Total Quantity Sold
    plt.figure(figsize=(12, 7))
    sns.barplot(x=performance_by_variety.index, y=performance_by_variety['Total_Quantity_Sold'], palette='viridis')
    plt.title('Total Quantity Sold per Tomato Variety (kg)', fontsize=16)
    plt.xlabel('Tomato Variety', fontsize=12)
    plt.ylabel('Total Quantity Sold (kg)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Plot 2: Net Profit
    plt.figure(figsize=(12, 7))
    # We sort the data by net profit for this plot to make it more insightful
    profit_plot_data = performance_by_variety.sort_values(by='Total_Net_Profit', ascending=False)
    sns.barplot(x=profit_plot_data.index, y=profit_plot_data['Total_Net_Profit'], palette='plasma')
    plt.title('Total Net Profit per Tomato Variety (€)', fontsize=16)
    plt.xlabel('Tomato Variety', fontsize=12)
    plt.ylabel('Total Net Profit (€)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()

    print("\n[ANALYSIS COMPLETE]")

# This line ensures the script runs when you execute the file
if __name__ == "__main__":
    run_comprehensive_analysis()