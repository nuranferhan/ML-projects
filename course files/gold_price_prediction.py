import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Data Collection and Processing

# Load the gold price dataset into a Pandas DataFrame
gold_price_data = pd.read_csv('gold price dataset.csv')

# Display the first 5 rows of the dataset
print(gold_price_data.head())

# Display the last 5 rows of the dataset
print(gold_price_data.tail())

# Get the shape of the dataset (rows and columns)
print(gold_price_data.shape)

# Display the basic information about the dataset
gold_price_data.info()

# Check for missing values in the dataset
print(gold_price_data.isnull().sum())

# Display the statistical summary of the dataset
print(gold_price_data.describe())

# Correlation Analysis

# Calculate the correlation matrix
correlation_matrix = gold_price_data.corr()

# Create a heatmap to visualize the correlation
plt.figure(figsize=(8, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.1f', cmap='Blues', annot_kws={'size': 8}, square=True, cbar=True)
plt.show()

# Display the correlation values of the 'GLD' column
print(correlation_matrix['GLD'])

# Plot the distribution of 'GLD' prices
sns.distplot(gold_price_data['GLD'], color='green')
plt.show()

# Splitting Features and Target Variable

# Define the feature matrix (X) and the target variable (Y)
X = gold_price_data.drop(columns=['Date', 'GLD'])
Y = gold_price_data['GLD']

# Print the feature matrix and target variable
print(X.head())
print(Y.head())

# Splitting the dataset into Training and Test sets

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Model Training: Random Forest Regressor

# Initialize the Random Forest Regressor model with 100 estimators
rf_regressor = RandomForestRegressor(n_estimators=100)

# Train the model using the training data
rf_regressor.fit(X_train, Y_train)

# Model Evaluation

# Make predictions on the test data
predictions = rf_regressor.predict(X_test)

# Display the predictions
print(predictions)

# Calculate the R-squared error of the model
r2_error = metrics.r2_score(Y_test, predictions)
print(f"R-squared error: {r2_error}")

# Visualizing the Actual vs Predicted Values

# Convert the test target values to a list for plotting
Y_test_list = list(Y_test)

# Plot the actual vs predicted values
plt.plot(Y_test_list, color='blue', label='Actual Value')
plt.plot(predictions, color='green', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of Values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()



