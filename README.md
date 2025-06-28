# Retail_PBM (Profitability & Predictive Behavior Model)

## Project Summary

This project is a dual-purpose data analysis tool for a retail environment, using tomato sales as a case study. It functions as both a **comprehensive profitability dashboard** and a **machine learning pipeline** that trains a model to predict future sales.

The primary goal is to provide a 360-degree view of the business, answering both "How did we perform?" and "What do we need to prepare for what might happen next?".

---

## Key Features

The project is currently divided into two main functional areas:

### 1. Profitability Analysis Dashboard

This module provides a detailed report on business performance based on historical sales data.

- **Financial Metrics:** Automatically calculates KPIs like Revenue, Gross Profit, and Net Profit after accounting for the cost of unsold stock (waste).
- **Time-Based Reporting:** Generates clear monthly and weekly financial summaries.
- **Product Performance Analysis:** Identifies top/bottom selling products by quantity and profitability.
- **Data Visualization:** Produces bar charts to visually compare product performance.

---

## Part 2: Sales Prediction Model

This module uses the historical data to train a `RandomForestRegressor` model to forecast future sales volume (in kg) for each tomato variety.

### Model Performance & Insights

After training the model on 80% of the data and testing it on a hidden 20% set, we obtained the following results:

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **R-squared (R²)** | `0.87` | The model successfully explains **87%** of the variance in daily sales. This indicates a very strong fit. |
| **Mean Absolute Error (MAE)** | `2.32 kg` | On average, the model's prediction is off by **±2.32 kg** from the actual sales figure. This is our real-world margin of error. |
| **Out-of-Bag (OOB) Score** | `0.96` | An internal model validation score. This high value confirms that the model generalizes well and is not "memorizing" the data. |

### The "Secret" of the Model: Feature Importance

We asked the model which clues (features) it found most important for making its predictions. The results reveal the core logic of our sales patterns:

1.  **`sales_lag_1_day` (Importance: 77.0%)**
    - **Insight:** This is the undisputed champion. **Yesterday's sales volume is the single most powerful predictor of today's sales.** This confirms that sales have strong day-to-day momentum.

2.  **`day_of_week` (Importance: 16.4%)**
    - **Insight:** The day of the week is the second most critical factor. This proves the model has learned a clear **weekly sales cycle** (e.g., sales patterns on weekends are different from weekdays).

3.  **Other Features (e.g., `Variety_Ensalada`, `day_of_month`)**
    - **Insight:** Specific product types and the day within the month provide the remaining ~7% of the information, helping the model make finer adjustments to its predictions.

**Conclusion:** The model's "secret" is that this is a heavily time-dependent problem. Its success relies on understanding the powerful influence of **recent history** and **weekly seasonality**.

---

## How to Run

1.  Ensure you have Python and the required libraries installed: `pip install pandas matplotlib seaborn scikit-learn`.
2.  Make sure your sales data is in the `sales_data.csv` file.
3.  Run the main script from your terminal:
    ```bash
    python predictive_model.py
    ```

## Next Steps

- **Prediction Function:** Create a function that uses the trained model to forecast sales for a specific future date.
- **Actionable Recommendations:** Translate the numerical prediction into a human-readable business suggestion (e.g., "Recommendation: Order X kg of 'Tomato Pera' for tomorrow").