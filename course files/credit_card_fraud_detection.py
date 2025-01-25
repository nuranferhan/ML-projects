import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the dataset into a Pandas DataFrame
credit_data = pd.read_csv('credit_data.csv')

# Displaying the first five rows of the dataset
print(credit_data.head())

# Displaying the last five rows of the dataset
print(credit_data.tail())

# Dataset information (including column types and non-null counts)
credit_data.info()

# Checking for missing values in each column
print(credit_data.isnull().sum())

# Distribution of normal and fraudulent transactions
print(credit_data['Class'].value_counts())

# Dataset Overview:
# 0 --> Normal Transaction
# 1 --> Fraudulent Transaction

# Splitting the dataset into legitimate and fraudulent transactions
legit_data = credit_data[credit_data['Class'] == 0]
fraud_data = credit_data[credit_data['Class'] == 1]

# Displaying the shape of both subsets
print(f"Legitimate Transactions: {legit_data.shape}")
print(f"Fraudulent Transactions: {fraud_data.shape}")

# Descriptive statistics for the 'Amount' feature in both subsets
print("Legitimate Transactions - Amount Stats:")
print(legit_data['Amount'].describe())

print("Fraudulent Transactions - Amount Stats:")
print(fraud_data['Amount'].describe())

# Comparing the mean values of features grouped by 'Class'
print(credit_data.groupby('Class').mean())

# Under-sampling: Balancing the dataset by selecting an equal number of fraudulent and legitimate transactions
legit_sampled = legit_data.sample(n=492)

# Combining the sampled legitimate data with fraudulent data
balanced_data = pd.concat([legit_sampled, fraud_data], axis=0)

# Displaying the first and last five rows of the new dataset
print(balanced_data.head())
print(balanced_data.tail())

# Checking the class distribution in the new balanced dataset
print(balanced_data['Class'].value_counts())

# Comparing the mean values of features grouped by 'Class' in the balanced dataset
print(balanced_data.groupby('Class').mean())

# Splitting the features and target variable
X_features = balanced_data.drop(columns='Class')
y_target = balanced_data['Class']

# Displaying the features and target
print(X_features.head())
print(y_target.head())

# Splitting the dataset into training and testing sets
X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X_features, y_target, test_size=0.2, stratify=y_target, random_state=2)

# Displaying the shapes of the training and test datasets
print(f"Features Shape: {X_features.shape}, Training Shape: {X_train_data.shape}, Test Shape: {X_test_data.shape}")

# Model Training: Logistic Regression
logistic_model = LogisticRegression()

# Training the Logistic Regression model with the training data
logistic_model.fit(X_train_data, y_train_data)

# Model Evaluation: Accuracy Score

# Evaluating accuracy on the training data
train_predictions = logistic_model.predict(X_train_data)
train_accuracy = accuracy_score(y_train_data, train_predictions)
print(f"Training Accuracy: {train_accuracy}")

# Evaluating accuracy on the test data
test_predictions = logistic_model.predict(X_test_data)
test_accuracy = accuracy_score(y_test_data, test_predictions)
print(f"Test Accuracy: {test_accuracy}")


