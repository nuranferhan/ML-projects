import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# Load the Boston Housing Dataset
boston_data = sklearn.datasets.load_boston()

# Print the dataset details
print(boston_data)

# Convert the dataset into a pandas DataFrame
df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)

# Display the first 5 rows of the DataFrame
df.head()

# Add the target variable (price) to the DataFrame
df['price'] = boston_data.target

# Display the first 5 rows after adding the target column
df.head()

# Check the dimensions of the DataFrame
df.shape

# Check for missing values in the DataFrame
df.isnull().sum()

# Get statistical summary of the dataset
df.describe()

# Calculate the correlation between features
correlation_matrix = df.corr()

# Create a heatmap to visualize the correlation
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.1f', cmap='Blues', square=True, cbar=True, annot_kws={'size': 8})

# Separate features (X) and target variable (Y)
X_data = df.drop(columns='price', axis=1)
Y_data = df['price']

# Display features and target variable
print(X_data)
print(Y_data)

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=2)

# Print the shapes of the datasets
print(f"Shape of X: {X_data.shape}, X_train: {X_train.shape}, X_test: {X_test.shape}")

# Initialize the XGBoost Regressor model
xgb_model = XGBRegressor()

# Train the model using the training data
xgb_model.fit(X_train, Y_train)

# Predict on the training data
train_predictions = xgb_model.predict(X_train)

# Print the predictions on the training data
print(train_predictions)

# Calculate R-squared error for training data
train_r2_score = metrics.r2_score(Y_train, train_predictions)

# Calculate Mean Absolute Error for training data
train_mae = metrics.mean_absolute_error(Y_train, train_predictions)

# Print evaluation metrics for training data
print(f"Training Data - R squared error: {train_r2_score}")
print(f"Training Data - Mean Absolute Error: {train_mae}")

# Visualize the actual vs predicted prices on the training data
plt.scatter(Y_train, train_predictions)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices (Training Data)")
plt.show()

# Predict on the test data
test_predictions = xgb_model.predict(X_test)

# Calculate R-squared error for test data
test_r2_score = metrics.r2_score(Y_test, test_predictions)

# Calculate Mean Absolute Error for test data
test_mae = metrics.mean_absolute_error(Y_test, test_predictions)

# Print evaluation metrics for test data
print(f"Test Data - R squared error: {test_r2_score}")
print(f"Test Data - Mean Absolute Error: {test_mae}")








