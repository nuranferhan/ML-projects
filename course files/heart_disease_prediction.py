import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data Collection and Processing

# Load the heart disease dataset into a pandas DataFrame
heart_disease_data = pd.read_csv('data.csv')

# Display the first five rows of the dataset
heart_disease_data.head()

# Display the last five rows of the dataset
heart_disease_data.tail()

# Get the number of rows and columns
print(heart_disease_data.shape)

# Display general information about the dataset
heart_disease_data.info()

# Check for missing values in the dataset
print(heart_disease_data.isnull().sum())

# Show statistical summary of the dataset
heart_disease_data.describe()

# Check the distribution of the target variable
print(heart_disease_data['target'].value_counts())

# Mapping of the target variable
# 1 --> Defective Heart
# 0 --> Healthy Heart

# Splitting the dataset into features (X) and target (Y)
X = heart_disease_data.drop(columns='target')
Y = heart_disease_data['target']

# Display the feature set and target variable
print(X.head())
print(Y.head())

# Splitting the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Display the shapes of the training and test datasets
print(f"X shape: {X.shape}, X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# Model Training: Logistic Regression

# Initialize the Logistic Regression model
log_reg_model = LogisticRegression()

# Train the model using the training data
log_reg_model.fit(X_train, Y_train)

# Model Evaluation: Accuracy Score

# Evaluate accuracy on the training data
train_predictions = log_reg_model.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_predictions)
print(f"Accuracy on Training data: {train_accuracy}")

# Evaluate accuracy on the test data
test_predictions = log_reg_model.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_predictions)
print(f"Accuracy on Test data: {test_accuracy}")

# Building a Predictive System

# Sample input data for prediction
input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)

# Convert the input data to a numpy array
input_data_array = np.asarray(input_data)

# Reshape the input data for prediction (as we are predicting for a single instance)
input_data_reshaped = input_data_array.reshape(1, -1)

# Predict the result using the trained model
prediction = log_reg_model.predict(input_data_reshaped)

# Display the prediction
if prediction[0] == 0:
    print("The person does not have heart disease.")
else:
    print("The person has heart disease.")




