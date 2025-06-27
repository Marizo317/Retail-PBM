# Retail_PBM - Predictive Behavior Model
# Steps 1 & 2: Data Loading and Exploratory Analysis

# --- 1. IMPORT LIBRARIES ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("--- Retail_PBM Project Initialized ---")

# --- 2. DATA LOADING AND NORMALIZATION ---
filename = 'sales_data.csv'
spanish_to_english_map = {
    'Fecha': 'Date', 'Variedad': 'Variety',
    'Cantidad_Vendida_kg': 'Quantity_Sold_kg',
    'Precio_por_kg': 'Price_per_kg', 'Coste_por_kg': 'Cost_per_kg'
}

try:
    df = pd.read_csv(filename)
    print(f"\nSuccess! File '{filename}' was loaded.")
    df.rename(columns=spanish_to_english_map, inplace=True)
except FileNotFoundError:
    print(f"\nERROR: Could not find the file '{filename}'.")
    print("Please make sure the script and the CSV file are in the same folder.")
    exit() # Exit the script if data is not found

# --- 3. EXPLORATORY DATA ANALYSIS (EDA) ---
print("\n--- Step 2: Exploratory Data Analysis ---")
df['Date'] = pd.to_datetime(df['Date'])
df['Revenue'] = df['Quantity_Sold_kg'] * df['Price_per_kg']
df['Total_Cost'] = df['Quantity_Sold_kg'] * df['Cost_per_kg']
df['Profit'] = df['Revenue'] - df['Total_Cost']
print("\nFinancial metrics (Revenue, Cost, Profit) calculated.")

performance_by_variety = df.groupby('Variety').agg(
    Total_Quantity_Sold=('Quantity_Sold_kg', 'sum'),
    Total_Profit=('Profit', 'sum')
).sort_values(by='Total_Quantity_Sold', ascending=False)

top_3_best_sellers_by_qty = performance_by_variety.head(3)
top_3_worst_sellers_by_qty = performance_by_variety.tail(3)
top_3_most_profitable = performance_by_variety.sort_values(by='Total_Profit', ascending=False).head(3)

print("\n🏆 Top 3 Best-Sellers (by Quantity):")
print(top_3_best_sellers_by_qty['Total_Quantity_Sold'])
print("\n📉 Top 3 Worst-Sellers (by Quantity):")
print(top_3_worst_sellers_by_qty['Total_Quantity_Sold'])
print("\n💰 Top 3 Most Profitable (by Total Profit):")
print(top_3_most_profitable['Total_Profit'])

# --- 4. VISUALIZATION ---
print("\nGenerating visualizations...")
sns.set_style("whitegrid")
plt.figure(figsize=(12, 7))
sns.barplot(x=performance_by_variety.index, y=performance_by_variety['Total_Quantity_Sold'], palette='viridis')
plt.title('Total Quantity Sold per Tomato Variety (kg)', fontsize=16)
plt.xlabel('Tomato Variety', fontsize=12)
plt.ylabel('Total Quantity Sold (kg)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.figure(figsize=(12, 7))
sns.barplot(x=performance_by_variety.index, y=performance_by_variety['Total_Profit'], palette='plasma')
plt.title('Total Profit per Tomato Variety (€)', fontsize=16)
plt.xlabel('Tomato Variety', fontsize=12)
plt.ylabel('Total Profit (€)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("\nAnalysis complete. Close the plot windows to finish.")