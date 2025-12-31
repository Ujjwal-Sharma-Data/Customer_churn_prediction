import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ------------------------------------------------------------------------------------------------
# 1. DEFINE CUSTOM CLASS (MUST BE HERE EXACTLY AS TRAINED)
# ------------------------------------------------------------------------------------------------
# This class definition is required for joblib to reconstruct the pipeline
class FeatureBinner(BaseEstimator, TransformerMixin):
    def __init__(self):  
        pass
    def fit(self, X, y=None):  
        return self

    def transform(self, X):  
        X = X.copy()
        # tenure bins
        if 'tenure' in X.columns:
            X['tenure_bin'] = pd.cut(       
                X['tenure'],
                bins=[0, 10, 60, float('inf')], # Adjusted max to float('inf') for safety
                labels=['new_customers','existing_customers','old_customers'], 
                right=True,
                include_lowest=True
            )
        # MonthlyCharges bins
        if 'MonthlyCharges' in X.columns:
            X['MonthlyCharges_bin'] = pd.cut(
                X['MonthlyCharges'],
                bins=[0, 30, 65, float('inf')], # Adjusted max to float('inf') for safety
                labels=['low_charges','medium_charges','high_charges'], 
                right=True,
                include_lowest=True
            )
        return X

# ------------------------------------------------------------------------------------------------
# 2. APP SETUP & MODEL LOADING
# ------------------------------------------------------------------------------------------------
st.set_page_config(page_title="Churn Prediction App", page_icon="📉", layout="centered")

@st.cache_resource
def load_model():
    try:
        return joblib.load('churn_final_model_pipeline_2.joblib')
    except FileNotFoundError:
        return None

pipeline = load_model()

# ------------------------------------------------------------------------------------------------
# 3. USER INTERFACE
# ------------------------------------------------------------------------------------------------
st.title("📉 Customer Churn Predictor")
st.write("Enter customer details below to predict the likelihood of churn.")

if pipeline is None:
    st.error("⚠️ Model file 'churn_model_pipeline.joblib' not found. Please ensure it is in the same directory.")
    st.stop()

# Form for user input
with st.form("prediction_form"):
    st.subheader("Customer Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=65.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=1000.0)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    with col2:
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        # Add other fields required by your model here with default values if not critical for UI
        # For example:
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

    # Hidden/Default fields (Your model expects these columns to exist, even if we don't ask the user)
    # Adjust these based on the exact columns your model was trained on
    defaults = {
        'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
        'PhoneService': 'Yes', 'MultipleLines': 'No', 'OnlineSecurity': 'No',
        'OnlineBackup': 'No', 'DeviceProtection': 'No', 'StreamingTV': 'No', 'StreamingMovies': 'No'
    }

    submit_btn = st.form_submit_button("🚀 Predict Churn Risk")

# ------------------------------------------------------------------------------------------------
# 4. PREDICTION LOGIC
# ------------------------------------------------------------------------------------------------
if submit_btn:
    try:
        # 1. Create DataFrame from inputs
        input_dict = {
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'Contract': contract,
            'PaymentMethod': payment_method,
            'InternetService': internet_service,
            'PaperlessBilling': paperless,
            'TechSupport': tech_support
        }
        # Merge with defaults
        input_dict.update(defaults)
        
        input_df = pd.DataFrame([input_dict])

        # 2. Get Prediction
        prediction = pipeline.predict(input_df)[0]
        prediction_proba = pipeline.predict_proba(input_df)[0][1]

        # 3. Display Results
        st.divider()
        st.subheader("Prediction Result")
        
        col_metric1, col_metric2 = st.columns(2)
        
        with col_metric1:
            st.metric(label="Churn Probability", value=f"{prediction_proba:.2%}")
        
        with col_metric2:
            if prediction_proba > 0.4: # You can adjust this threshold
                st.error("⚠️ **High Churn Risk**")
                st.write("Customer is likely to cancel.")
            else:
                st.success("✅ **Low Churn Risk**")
                st.write("Customer is likely to stay.")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.write("Please check if your input DataFrame columns match the trained model columns exactly.")
