# Telecom customer churn analysis 📊

## Project overview

This project analyses customer churn in the telecommunications industry to identify patterns in customer behaviour, understand factors associated with churn and highlight opportunities to improve customer retention.

Using Python and exploratory data analysis, the project examines customer demographics, service usage, contract types, billing information and customer tenure. The findings are translated into business insights that can support customer retention strategies and data driven decision making.

## Objectives

- Assess data quality and identify potential inconsistencies.
- Explore customer churn patterns across different customer segments.
- Identify customer characteristics and service features associated with higher churn.
- Analyse customer tenure and its relationship with retention.
- Explore customer lifetime value and identify higher risk customer profiles.
- Translate analytical findings into actionable business insights.

## Tools and technologies 🐍

- **Python:** Data analysis and manipulation
- **Pandas:** Data cleaning, transformation and exploratory analysis
- **NumPy:** Numerical operations
- **Matplotlib:** Data visualisation
- **Jupyter Notebook:** Interactive analysis and documentation

## Project workflow

### 1. Data quality assessment

Reviewed the dataset structure, data types, missing values, duplicate records and distinct values. Identified inconsistent values in the total charges column and investigated discrepancies between monthly charges, tenure and recorded total charges.

### 2. Exploratory data analysis

Examined customer demographics, subscriptions, payment methods, contract types and service features to understand how these characteristics relate to customer churn.

### 3. Customer segmentation

Grouped customers by tenure to compare churn across different stages of the customer lifecycle. Analysed churn rates across customer groups to identify segments that may require additional retention efforts.

### 4. Customer lifetime value analysis

Explored customer lifetime value using monthly charges and tenure as a simplified measure of accumulated customer revenue. Compared the results between customers who churned and those who remained.

### 5. Customer risk analysis

Developed an exploratory, rule based customer risk score using four indicators: month to month contracts, short tenure, lack of technical support and lack of online security. Used the score to investigate potential retention risk factors.

## Key findings

- **Contract type:** Customers on month to month contracts had a churn rate of approximately 42.7%, compared with 11.3% for one year contracts and 2.8% for two year contracts.
- **Customer tenure:** Customers with shorter tenure were more likely to churn. The churn rate was approximately 47.4% among customers with a tenure of 12 months or less, compared with 9.5% among long term customers with more than 48 months of tenure.
- **Additional services:** Customers without online security or technical support showed substantially higher churn rates than customers with these services.
- **Customer profiles:** The combination of a month to month contract, electronic cheque payment and fibre optic internet had an observed churn rate of approximately 60.4%.
- **Data quality:** Identified 11 records with non numeric total charges values and investigated inconsistencies between recorded charges and an estimate based on monthly charges and tenure.

These findings highlight customer groups that may benefit from targeted retention initiatives, improved onboarding and further investigation into service adoption and customer experience.

## Repository structure

```text
telecom-customer-churn-analysis/
├── notebooks/
│   └── telecom_customer_churn_analysis.ipynb
├── screenshots/
└── README.md
```

## How to explore the project

Open the Jupyter Notebook to review the analysis, Python code, calculations and findings. The `screenshots` folder contains selected visualisations from the analysis.

The notebook uses Python libraries including Pandas, NumPy and Matplotlib.

## Notes and limitations

The dataset is not included in this repository. Running the notebook from beginning to end requires access to the original dataset and an appropriate local file path.

Customer lifetime value is estimated using monthly charges multiplied by tenure and should be interpreted as a simplified analytical measure rather than a complete measure of profitability. The customer risk score is an exploratory, rule based approach and is not a trained machine learning model.

## Skills demonstrated

Exploratory data analysis, data quality assessment, data cleaning, customer segmentation, churn analysis, KPI interpretation, data visualisation and translating data into business insights.
