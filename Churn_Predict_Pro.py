import streamlit as st
import pandas as pd
import numpy as np
import joblib  
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model

# Loading the trained model
best_model = load_model('best_model.h5')  

# Loading the scaler
scaler = joblib.load('scaler.pkl')

# Function to make predictions and obtain confidence scores

# Function to make predictions and obtain confidence scores
def predict_churn(features, label_encoders):
    # Transforming categorical columns using the loaded label encoders
    for feature, encoder in label_encoders.items():
        if feature in features:
            features[feature] = encoder.transform([features[feature]])[0]

    # Converting features to a NumPy array with the correct data type
    features_array = np.array(features, dtype=np.float32)

    # Making predictions using the loaded model
    raw_prediction = best_model.predict(features_array)
    # Assuming binary classification, extracting the probability of class 1
    probability = raw_prediction[0]

    # Converting the raw prediction into a binary prediction
    prediction = 1 if probability >= 0.5 else 0

    return prediction, probability * 100


# Streamlit app
def main():
    st.title('Churn Predict Pro')

    # Adding UI elements for user input
    monthly_charges = st.slider('Monthly Charges', min_value=0, max_value=100, value=50)
    total_charges = st.slider('Total Charges', min_value=0, max_value=5000, value=2500)
    tenure = st.slider('Tenure (months)', min_value=0, max_value=100, value=12)

    # Creating a dictionary to store LabelEncoders for each categorical feature
    label_encoders = {}

    # Categorical features
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two years'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
    tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
    paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    partner = st.selectbox('Partner', ['Yes', 'No'])
    multiple_lines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
    device_protection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
    dependents = st.selectbox('Dependents', ['Yes', 'No'])
    senior_citizen = st.selectbox('Senior Citizen', ['Yes', 'No'])
    streaming_movies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
    streaming_tv = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
    phone_service = st.selectbox('Phone Service', ['Yes', 'No'])

    # Adding LabelEncoders to the dictionary and fitting them with training data
    for feature in ['Contract', 'PaymentMethod', 'OnlineSecurity', 'TechSupport', 'Gender', 'OnlineBackup', 
                'PaperlessBilling', 'InternetService', 'Partner', 'MultipleLines', 'DeviceProtection', 
                'Dependents', 'SeniorCitizen', 'StreamingMovies', 'StreamingTV', 'PhoneService']:
        label_encoder = LabelEncoder()
        label_encoder.fit(features[feature])
        label_encoders[feature] = label_encoder


    # Creating a feature dataframe for prediction
    features = pd.DataFrame({
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'tenure': [tenure],
        'Contract': [contract],
        'PaymentMethod': [payment_method],
        'OnlineSecurity': [online_security],
        'TechSupport': [tech_support],
        'gender': [gender],
        'OnlineBackup': [online_backup],
        'PaperlessBilling': [paperless_billing],
        'InternetService': [internet_service],
        'Partner': [partner],
        'MultipleLines': [multiple_lines],
        'DeviceProtection': [device_protection],
        'Dependents': [dependents],
        'SeniorCitizen': [senior_citizen],
        'StreamingMovies': [streaming_movies],
        'StreamingTV': [streaming_tv],
        'PhoneService': [phone_service]
    })

    # Making a prediction and obtain confidence score
    prediction, confidence = predict_churn(features, label_encoders)

    # Displaying the prediction result and confidence score
    st.subheader('Prediction Result:')
    if prediction == 1:
        st.write('The customer is likely to churn with a confidence score of {:.2f}%.'.format(confidence))
    else:
        st.write('The customer is likely to stay with a confidence score of {:.2f}%.'.format(100 - confidence))

if __name__ == '__main__':
    main()
