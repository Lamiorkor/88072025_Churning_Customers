import streamlit as st
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Load the trained model
model = load_model('best_model.h5')

# Replace the existing code for loading the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
def main():
    st.title('Churn Prediction App')

    # Collect user input
    tenure = st.slider('Tenure', 0, 70, 20)
    monthly_charges = st.slider('Monthly Charges', 0.0, 200.0, 100.0)
    total_charges = st.slider('Total Charges', 0.0, 5000.0, 2500.0)
    contract = st.selectbox('Contract', ['month-to-month', 'one year', 'Two years'])
    online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Credit card (automatic)'])
    tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
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
        prediction_proba = model.predict(scaled_input)[0]
        prediction = 1 if prediction_proba >= 0.5 else 0  # Convert probability to binary prediction

        # Display the result
        st.write(f'Churn Prediction: {prediction} with Confidence: {prediction_proba:.2%}')

if __name__ == '__main__':
    main()