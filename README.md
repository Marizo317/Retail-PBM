# Retail_PBM (Profitability & Predictive Behavior Model)

## Project Summary

This project is a dual-purpose data analysis tool for a retail environment, using tomato sales as a case study. It functions as both a **comprehensive profitability dashboard** and a **complete machine learning pipeline** that trains a model and uses it to predict future sales.

The primary goal is to provide a 360-degree view of the business, answering "How did we perform?" with the analysis dashboard, and "What should we stock for tomorrow?" with actionable, data-driven recommendations.

---

## Part 1: Profitability Analysis Dashboard

This module provides a detailed report on business performance based on historical sales data.

- **Financial Metrics:** Automatically calculates KPIs like Revenue, Gross Profit, and Net Profit after accounting for the cost of unsold stock (waste).
- **Time-Based Reporting:** Aggregates financial data to generate clear monthly and weekly performance summaries.
    - **Key Feature:** The script analyzes Net Profit and **returns a clear status (e.g.,  PROFITABLE /  NOT PROFITABLE)** for each period.
- **Product Performance Analysis:** Groups data by product variety to generate a detailed report, including:
    - A full performance table with sales, gross profit, waste cost, and net profit per variety.
    - The explicit **Top 3 & Bottom 3** best-selling products by quantity.
    - The **Top 3** most profitable products by net profit.
- **Data Visualization:** Produces bar charts to visually compare product performance.

---

## Part 2: Sales Prediction Pipeline

This module uses the historical data to train a `RandomForestRegressor` model and then uses it to forecast future sales volume.

### Model Performance & Insights

After training, the model's performance was evaluated on a hidden test set:

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **R-squared (R²)** | `0.87` | The model successfully explains **87%** of the variance in daily sales. This indicates a very strong fit. |
| **Mean Absolute Error (MAE)** | `2.32 kg` | On average, the model's prediction is off by **±2.32 kg** from the actual sales figure. This is our real-world margin of error. |

The model's most important features were **yesterday's sales (`sales_lag_1_day`)** and the **day of the week (`day_of_week`)**, confirming that sales are driven by recent momentum and weekly seasonality.

### Forecasting & Business Recommendations

The final step in the pipeline uses the trained model to generate concrete, actionable recommendations for the next business day.

- **Process:** The script forecasts sales for the day following the last date in the dataset. It does this for each product variety by constructing the necessary features (last day's sales, day of the week, etc.) and feeding them to the model.
- **Output:** The result is a clear, easy-to-read table that provides a specific stock recommendation for each product.

---

## How to Run

1.  Ensure you have Python and the required libraries installed: `pip install pandas matplotlib seaborn scikit-learn`.
2.  Make sure your sales data is in the `sales_data.csv` file.
3.  Run the main script from your terminal:
    ```bash
    python predictive_model.py
    ```
4.  Review the full report in the terminal, which includes profitability analysis and the final purchase recommendations.

## Future Improvements

With the core functionality complete, future enhancements could include:
- **Hyperparameter Tuning:** Fine-tuning the `RandomForestRegressor` model to potentially improve its accuracy.
- **Advanced Feature Engineering:** Incorporating external data like public holidays, weather forecasts, or promotional events.
- **Intelligent Stock Management:** Implementing a system based on batch/lot expiration dates to trigger automatic offers and reduce waste.
- **Deployment:** Packaging the script into a simple web application (using Streamlit or Flask) or an automated daily email report.