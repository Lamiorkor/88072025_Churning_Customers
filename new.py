import streamlit as st
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load the trained model
model = load_model('best_model.h5')

scaler = joblib.load('scaler.pkl')

# Streamlit app
def main():
    st.title('Churn Predict Pro')

    # Collect user input
    tenure = st.slider('Tenure', 0, 70, 20)
    monthly_charges = st.slider('Monthly Charges', 0.0, 200.0, 100.0)
    total_charges = st.slider('Total Charges', 0.0, 5000.0, 2500.0)
    contract = st.selectbox('Contract', ['month-to-month', 'one year', 'Two years'])
    online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Credit card (automatic)'])
    tech_support = st.selectbox('Tech Support',['No', 'Yes', 'No internet service'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
    gender = st.radio('Gender', ['Male', 'Female'])
    paperless_billing = st.radio('Paperless Billing', ['No', 'Yes'])
    partner = st.radio('Partner', ['No', 'Yes'])
    multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
    device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
    dependents = st.radio('Dependents', ['No', 'Yes'])
    senior_citizen = st.radio('Senior Citizen', ['No', 'Yes'])
    streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
    streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    phone_service = st.selectbox('Phone Service', ['No', 'Yes'])

    # Make a prediction
    if st.button('Predict Churn'):
        # Transform user input
        user_input = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'Contract': [contract],
            'OnlineSecurity': [online_security],
            'PaymentMethod': [payment_method],
            'TechSupport': [tech_support],
            'InternetService': [internet_service],
            'OnlineBackup': [online_backup],
            'gender': [gender],
            'PaperlessBilling': [paperless_billing],
            'Partner': [partner],
            'MultipleLines': [multiple_lines],
            'DeviceProtection': [device_protection],
            'Dependents': [dependents],
            'SeniorCitizen': [senior_citizen],
            'StreamingMovies': [streaming_movies],
            'StreamingTV': [streaming_tv],
            'PhoneService': [phone_service]
        })

        # Encode categorical variables
        label_encoder = LabelEncoder()
        categorical_columns = ['Contract', 'OnlineSecurity', 'PaymentMethod', 'TechSupport', 'InternetService', 'OnlineBackup', 'gender',
                               'PaperlessBilling', 'Partner', 'MultipleLines', 'DeviceProtection', 'Dependents', 'SeniorCitizen',
                               'StreamingMovies', 'StreamingTV', 'PhoneService']

        for column in categorical_columns:
            user_input[column] = label_encoder.fit_transform(user_input[column])

        # Scale the input
        scaled_input = scaler.transform(user_input.values)

        # Make a prediction
        # Make a prediction
        predictions = model.predict(scaled_input)
        prediction_proba = predictions[0]  # Assuming the positive class is at index 0

        if prediction_proba >= 0.5:
            st.write(f"The customer will churn with a confidence score of {prediction_proba[0]:.2%}")
        else:
            st.write(f"The customer will not churn with a confidence score of {prediction_proba[0]:.2%}")

if __name__ == '__main__':
    main()
