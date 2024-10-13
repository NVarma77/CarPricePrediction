# Car Price Analyzer

## Project Overview

This project aims to predict car prices using various features from a car dataset. Using a supervised learning approach, specifically linear regression, we trained the model on data that included car prices. This model explores which features most significantly affect car prices, with the final goal of achieving high predictive accuracy.

## Prerequisites

Before running the notebook, please download the following:
1. `CarPrice_Assignment.csv` – The dataset file.
2. `CarPriceAnalyzer.ipynb` – The Jupyter notebook file.

Ensure both files are in the same directory before launching the notebook.

## Thought Process and Approach

### 1. Data Exploration and Understanding
   - We began by examining the dataset to understand feature distributions and identify potential trends.
   - We focused on numerical features to generate a correlation matrix and visualize it with a heatmap. Features showing high correlation were analyzed further for collinearity.

### 2. Cleaning the Data
   - For the 'CarName' feature, we extracted the car company name by stripping all terms beyond the first word. This resulted in a new feature: `Car_Company`.
   - A frequency check (`value_counts`) was conducted to identify any discrepancies in company names. Detected spelling errors were corrected to ensure data consistency.

### 3. Data Preparation
   - Several categorical features were still present, including numerical representations as strings (e.g., "One," "Two," "Three").
   - A lambda function translated these words into numbers. Dummy variables were created for other categorical features, and the first dummy variable was dropped to prevent collinearity.

### 4. Model Building
   - Data was split into training and testing sets with a 70/30 ratio. Standardization was applied to ensure consistent feature scaling.
   - An initial linear regression model was built with all features. Using Recursive Feature Elimination (RFE), we reduced the feature count to 15.
   - An Ordinary Least Squares (OLS) regression was performed on these 15 features. Features with Variance Inflation Factor (VIF) > 10 or p-values > 0.05 were removed to enhance model efficiency.

### 5. Linear Regression
   - The final model achieved an R-squared score of ~90%, indicating strong predictive power.

## Running the Notebook

To run the notebook:
1. Open `CarPriceAnalyzer.ipynb` in Jupyter Notebook or any compatible platform.
2. Execute each cell in sequence to observe the data processing, cleaning, model building, and evaluation stages.
3. Examine the output for model metrics and insights about significant features impacting car prices.

## Libraries Used
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `statsmodels`
- `sklearn`
