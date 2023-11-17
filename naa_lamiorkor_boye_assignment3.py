# -*- coding: utf-8 -*-
"""Naa Lamiorkor Boye_Assignment3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sKERY5F-HQz1j3ZII43uM0vhMk1aLNU4
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from tensorflow import keras
from keras.models import Model, load_model, save_model
from keras.layers import Input, Dense
!pip install scikeras
from scikeras.wrappers import KerasClassifier, BaseWrapper
from sklearn.metrics import accuracy_score, roc_auc_score

from google.colab import drive
drive.mount('/content/drive')

churn = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/CustomerChurn_dataset.csv')

churn.head()

churn.info()

# Getting the categorical columns
categorical = churn.select_dtypes(include=['object']).columns.tolist()
categorical

# Encoding the object columns using the LabelEncoder
label_encoder = LabelEncoder()
for column in categorical:
    churn[column] = label_encoder.fit_transform(churn[column])

churn.info()

# Defining independent and dependent variables
X = churn.drop(columns=['Churn','customerID'])
y = churn['Churn']

# Scaling the data
scaler = StandardScaler()
scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a Random Forest classifier
rfc = RandomForestClassifier(n_estimators=110,max_depth=20,criterion='entropy')

# Training the model
rfc.fit(X_train, y_train)

# Making predictions on the test set
y_pred = rfc.predict(X_test)

# Calculating accuracy for reference
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Getting feature importances
feature_importances = rfc.feature_importances_

# Creating a DataFrame to display feature importances
feature_importance_df = pd.DataFrame(
    {"Feature": X_train.columns, "Importance": feature_importances}
)

# Sorting features by importance
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

# Displaying the feature importance DataFrame
print("\nFeature Importance: \n")
print(feature_importance_df)

# Creating the model using a function
def create_model(neurons=10, activation='relu'):
    input_layer = Input(shape=(X_train.shape[1],))
    hidden_layer1 = Dense(neurons, activation=activation)(input_layer)
    hidden_layer2 = Dense(neurons, activation=activation)(hidden_layer1)
    hidden_layer3 = Dense(neurons, activation=activation)(hidden_layer2)
    hidden_layer4 = Dense(neurons, activation=activation)(hidden_layer3)
    output_layer = Dense(1, activation='sigmoid')(hidden_layer4)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Creating a KerasClassifier object
model = KerasClassifier(model=create_model, neurons=64, verbose=0)

# Defining the grid search parameters
param_grid = {
    'neurons': [64, 32, 16, 8],
    'epochs': [10, 15, 20],
    'batch_size': [16, 32, 64],
    'validation_split': [0.1, 0.2, 0.3]
}

# Using GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, y_train)

# Printing the best parameters and corresponding accuracy
print(f"Best Parameters: {grid_result.best_params_}")
print(f"Best Accuracy: {grid_result.best_score_}")

# Best parameters from the grid search
best_params = {'batch_size': 16, 'epochs': 20, 'neurons': 8, 'validation_split': 0.1}

# Instantiating the model with the best parameters
best_model = create_model(neurons=best_params['neurons'])

# Compiling the model
best_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model with the best parameters
best_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], validation_split=best_params['validation_split'])

# Making predictions on the test set
y_pred = best_model.predict(X_test)

# Converting predictions to binary (0 or 1)
y_pred_binary = (y_pred > 0.5).astype(int)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy: {accuracy}")

# Calculating ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC AUC Score: {roc_auc}")

# Saving the scaler
joblib.dump(scaler, 'scaler.pkl')

# Saving the model
best_model.save('best_model.h5')

from google.colab import files

# Download the saved model
files.download('scaler.pkl')
files.download('best_model.h5')