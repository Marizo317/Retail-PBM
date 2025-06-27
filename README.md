# Retail_PBM (Profitability & Predictive Behavior Model)

## Project Summary

This project is a dual-purpose data analysis tool for a retail environment, using tomato sales as a case study. It functions both as a **comprehensive profitability dashboard** and as the initial data pipeline for a **sales prediction model**.

The primary goal is to provide a 360-degree view of the business, answering both "Where can we adjust for better performance?" "How can we anticipate with the data obtained?" The data does not have to be repeated; past events do not guarantee that they will happen again, but they are the best tool for making increasingly accurate predictions

---

## Key Features

The project is currently divided into two main functional areas:

### 1. Profitability Analysis Dashboard

This module provides a detailed report on business performance based on historical sales data.

- **Financial Metrics:** Automatically calculates essential KPIs like Revenue, Gross Profit, Potential Waste (from unsold stock), and Net Profit.
- **Time-Based Reporting:** Aggregates financial data to generate clear monthly and weekly performance summaries.
- **Product Performance Analysis:** Groups data by product variety to identify:
    - Top & Bottom 3 best-sellers by quantity.
    - Top 3 most profitable products.
- **Data Visualization:** Generates clear bar charts to visually compare product performance in terms of total units sold and net profitability.

### 2. Predictive Modeling Pipeline (Data Preparation)

This module prepares the raw sales data and transforms it into a "model-ready" dataset, suitable for training a machine learning algorithm. This process is also known as Feature Engineering.

- **Time-Based Features:** Deconstructs dates into numerical features (e.g., day of the week, week of the year, month) to help the model learn cyclical patterns.
- **Lag Features:** Creates features based on past values (e.g., sales of a product from the previous day), which are highly indicative of future sales.
- **Categorical Encoding:** Converts text-based features like product variety into a numerical format (One-Hot Encoding) that machine learning models can understand.

---

## How to Run

1.  Ensure you have Python and the required libraries installed: `pip install pandas matplotlib seaborn`.
2.  Make sure your sales data is correctly formatted in the `sales_data.csv` file.
3.  Open a terminal inside the project folder and run the main script:
    ```bash
    python predictive_model.py
    ```
4.  Review the text-based reports in the terminal and the plot windows that appear.
