import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Data Collection and Preprocessing

# Load the dataset into a pandas DataFrame
loan_data = pd.read_csv('dataset.csv')

# Check the type of the dataset
print(type(loan_data))

# Display the first 5 rows of the dataset
print(loan_data.head())

# Get the shape of the dataset (number of rows and columns)
print(loan_data.shape)

# Display statistical measures of the dataset
print(loan_data.describe())

# Count the missing values in each column
print(loan_data.isnull().sum())

# Drop rows with missing values
loan_data.dropna(inplace=True)

# Verify that there are no missing values
print(loan_data.isnull().sum())

# Label encoding: Replace Loan_Status values 'N' and 'Y' with 0 and 1
loan_data.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)

# Display the first 5 rows after label encoding
print(loan_data.head())

# Display the distribution of 'Dependents' values
print(loan_data['Dependents'].value_counts())

# Replace '3+' in 'Dependents' column with 4
loan_data.replace(to_replace='3+', value=4, inplace=True)

# Display the updated distribution of 'Dependents'
print(loan_data['Dependents'].value_counts())

# Data Visualization

# Education vs Loan Status
sns.countplot(x='Education', hue='Loan_Status', data=loan_data)

# Marital Status vs Loan Status
sns.countplot(x='Married', hue='Loan_Status', data=loan_data)

# Convert categorical columns to numerical values
loan_data.replace({
    'Married': {'No': 0, 'Yes': 1},
    'Gender': {'Male': 1, 'Female': 0},
    'Self_Employed': {'No': 0, 'Yes': 1},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
    'Education': {'Graduate': 1, 'Not Graduate': 0}
}, inplace=True)

# Display the updated dataset
print(loan_data.head())

# Separate the features (X) and target (Y)
X_features = loan_data.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
Y_target = loan_data['Loan_Status']

# Display the features and target
print(X_features)
print(Y_target)

# Train-Test Split

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y_target, test_size=0.1, stratify=Y_target, random_state=2)

# Display the shapes of the splits
print(f"X shape: {X_features.shape}, X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# Model Training

# Initialize the Support Vector Machine classifier with a linear kernel
svm_classifier = svm.SVC(kernel='linear')

# Train the model with the training data
svm_classifier.fit(X_train, Y_train)

# Model Evaluation

# Predict on the training data
train_predictions = svm_classifier.predict(X_train)

# Calculate the accuracy score on the training data
train_accuracy = accuracy_score(Y_train, train_predictions)
print(f'Accuracy on training data: {train_accuracy}')

# Predict on the test data
test_predictions = svm_classifier.predict(X_test)

# Calculate the accuracy score on the test data
test_accuracy = accuracy_score(Y_test, test_predictions)
print(f'Accuracy on test data: {test_accuracy}')

# Predictive System

# Further steps for making predictions can be added here if needed
