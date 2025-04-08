import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

## Load the trained model
model = tf.keras.models.load_model("model.h5")

## Load the encoder and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# --- Streamlit UI ---
st.title("Customer Churn Prediction")

# Input fields
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider('Tenure', 0, 10)
num_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# --- Prepare input data ---
base_data = pd.DataFrame({
    'Credit Score': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Encode 'Geography' with one-hot
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Combine all columns
input_data = pd.concat([base_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Ensure feature alignment
expected_columns = list(scaler.feature_names_in_)
input_data = input_data.reindex(columns=expected_columns, fill_value=0)

# Scale input
input_data_scaled = scaler.transform(input_data)

# --- Make prediction ---
prediction_proba = model.predict(input_data_scaled)[0][0]

# --- Show result ---
if prediction_proba > 0.5:
    st.success(f"The customer is likely to churn. 🔻 (Probability: {prediction_proba:.2f})")
else:
    st.info(f"The customer is likely to stay. ✅ (Probability: {1 - prediction_proba:.2f})")
