# Retail_PBM (Predictive Behavior Model)

The primary goal of this project is to optimize retail profitability by directly addressing a key challenge: product waste. By analyzing historical sales data, this model identifies trends and forecasts future demand for perishable goods, such as tomatoes. 

These accurate forecasts enable smarter stock management, ensuring that purchases are aligned with expected sales. This process minimizes losses from unsold, expired products (waste margin adjustment) and prevents lost sales due to stockouts, ultimately leading to improved profit margins and optimized sales performance.

---

## Features

- **Universal Applicability:** While this demo uses tomato sales, the model's logic can be applied to **any product section** in a retail environment (e.g., bakery, dairy, butchery) to optimize stock and reduce waste.

- **Bilingual Data Ingestion:** The script is designed to seamlessly load `.csv` data files with headers in either **English or Spanish**, making it highly adaptable to different system configurations.

- **System Agnostic:** Easily integrates with **any environment or Point of Sale (POS) system** capable of exporting daily or transactional sales data to a `.csv` file.

- **Automated EDA & Financial Metrics:** Automatically calculates key metrics like Revenue, Cost, and Profit, and performs an exploratory data analysis to identify top and bottom-performing products.

- **Clear Data Visualization:** Generates easy-to-understand charts summarizing sales performance and profitability, allowing for quick business insights.

---

## Technologies Used

- **Python:** The core programming language.
- **Pandas:** For data manipulation and analysis.
- **Matplotlib & Seaborn:** For data visualization.
- **Scikit-learn:** (Will be used for the prediction model in future steps).

---

## How to Run the Project

1.  **Prerequisites:** Make sure you have Python and the required libraries installed:
    ```bash
    pip install pandas matplotlib seaborn
    ```
2.  **Clone the repository (or download the files).**
3.  **Navigate to the project folder** in your terminal:
    ```bash
    cd path/to/Retail_PBM
    ```
4.  **Execute the script:**
    ```bash
    python predictive_model.py
    ```
The script will print the analysis results in the console and display two plots.