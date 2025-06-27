# Retail_PBM (Profitability & Behavior Model)

## Project Overview

This project is a comprehensive analysis dashboard for the tomato department of a retail business. It processes daily sales and stock data from a CSV file to generate a full report on business performance, combining both financial profitability over time and individual product performance.

The main goal is to provide actionable insights to optimize purchasing, reduce waste, and improve overall profit margins by understanding what products are key sellers and which ones are most profitable.

## Key Features

- **Data Loading:** Imports sales data from a `sales_data.csv` file. Handles data with headers in both English and Spanish.
- **Financial Calculation:** Computes crucial metrics like Revenue, Gross Profit, Potential Waste (unsold stock), and an estimated Net Profit after accounting for waste costs.
- **Time-Based Analysis:** Generates a financial report aggregated by month and by week, allowing for performance tracking over time.
- **Product-Based Analysis:** Creates a detailed report on each tomato variety, identifying best/worst sellers by quantity and most/least profitable products.
- **Visualization:** Produces clear bar charts to visually compare the total quantity sold and the total net profit for each product variety.

## How to Run

1.  Ensure you have Python and the required libraries installed: `pip install pandas matplotlib seaborn`.
2.  Make sure your sales data is correctly formatted in the `sales_data.csv` file within the project folder.
3.  Open a terminal inside the project folder and run the main script:
    ```bash
    python predictive_model.py
    ```
4.  Review the text-based reports in the terminal and the plot windows that appear.