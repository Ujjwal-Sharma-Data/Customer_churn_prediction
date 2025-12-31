# Telecom Churn Prediction System
> **Live Demo:** [Click here to test the Churn Predictor](https://customerchurnprediction-mfeufgjrvtxxvcaqsn86qm.streamlit.app)

## Executive Summary
Customer Churn (attrition) is a critical problem for telecom companies, where acquiring a new customer costs **5-25x more** than retaining an existing one.

This project implements an end-to-end Machine Learning pipeline to identify at-risk customers. Unlike "black-box" approaches, this solution prioritizes **Recall (75%)**—ensuring we catch the vast majority of customers about to leave—while maintaining high interpretability for business stakeholders using **Logistic Regression**.

## Model Performance & ROI
The model was tuned to minimize False Negatives (missing a churner), as the cost of lost revenue far outweighs the cost of a retention discount.

| Metric | Score (Class 1 - Churn) | Business Interpretation |
| :--- | :--- | :--- |
| **Recall** | **0.75** | The model successfully catches **75%** of all customers who are actually leaving. |
| **Precision** | **0.52** | When the model flags a customer, it is correct 52% of the time. This is an acceptable trade-off to maximize retention reach. |
| **Accuracy** | **0.75** | Overall correctness across both classes. |
| **ROC-AUC** | **0.84** | Strong ability to distinguish between churners and non-churners. |

## Technical Architecture

### 1. Data Pipeline & Engineering
The dataset (7,043 rows) contained significant class imbalance (26% Churn) and required custom transformation:
* **Custom Feature Binner:** Developed a `FeatureBinner` transformer to handle non-linear relationships in `tenure` (e.g., binning 'new', 'existing' and 'old' customers) and `MonthlyCharges`('low', 'medium' and 'high' charges).
* **Handling Imbalance:** Applied **SMOTE** (Synthetic Minority Over-sampling) strictly within the cross-validation training folds using `ImbPipeline` to prevent data leakage.
* **Skewness Correction:** Applied Yeo-Johnson transformations to normalize `TotalCharges`.

### 2. Model Benchmarking Strategy
I implemented a randomized search harness to benchmark 5 distinct algorithms. The goal was to balance predictive power with business explainability.

| Algorithm | Performance | Verdict |
| :--- | :--- | :--- |
| **Logistic Regression** | **High Recall** | **Winner:** Best calibration and interpretability (feature coefficients). |
| **XGBoost** | High Accuracy | Prone to overfitting on this dataset size; harder to explain to non-technical teams. |
| **LightGBM** | High Accuracy | Fast training, but similar interpretability issues to XGBoost. |
| **Random Forest** | Medium | Good baseline, but probability scores were less calibrated than LogReg. |
| **Linear SVC** | Medium | Computationally expensive to derive probabilities (requires Platt scaling). |

**Why Logistic Regression?**
Business stakeholders need to know *why* a customer is churning. Logistic Regression allows us to state clearly: *"Being on a Month-to-Month contract increases churn odds by X%."* This actionable insight is often more valuable than a 1% gain in accuracy.

## Key Insights from EDA
1.  **The "Fiber Optic" Trap:** Customers with Fiber Optic internet are significantly more likely to churn than DSL users, identifying a potential quality of service issue.
2.  **Contract Lock-in:** Month-to-month contracts are the single biggest predictor of churn.
3.  **Payment Methods:** Electronic Check users show much higher attrition rates compared to those on automatic bank transfers.

## How to Run Locally

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/telco-churn-predictor.git](https://github.com/YOUR_USERNAME/telco-churn-predictor.git)
    cd telco-churn-predictor
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit App**
    ```bash
    streamlit run app.py
    ```
