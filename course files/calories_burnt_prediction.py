import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# Data Collection and Preparation

# Load the 'calories' and 'exercise' datasets from CSV files
calories = pd.read_csv('calories.csv')
exercise_data = pd.read_csv('/content/exercise.csv')

# Display the first 5 rows of each DataFrame
print(calories.head())
print(exercise_data.head())

# Merge the two DataFrames on columns (axis=1) to combine exercise data with the target 'Calories'
calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)

# Show the first few rows of the combined DataFrame
print(calories_data.head())

# Check the shape of the combined dataset (number of rows and columns)
print(f"Shape of the dataset: {calories_data.shape}")

# Display detailed information about the DataFrame
calories_data.info()

# Check for any missing values in the dataset
print(f"Missing Values:\n{calories_data.isnull().sum()}")

# Data Analysis

# Get some statistical summary of the dataset
print(calories_data.describe())

# Data Visualization

# Set the style for seaborn plots
sns.set()

# Visualize the distribution of 'Gender' using a count plot
sns.countplot(x='Gender', data=calories_data)
plt.title('Gender Distribution')
plt.show()

# Plot the distribution of the 'Age' column
sns.histplot(calories_data['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

# Plot the distribution of the 'Height' column
sns.histplot(calories_data['Height'], kde=True)
plt.title('Height Distribution')
plt.show()

# Plot the distribution of the 'Weight' column
sns.histplot(calories_data['Weight'], kde=True)
plt.title('Weight Distribution')
plt.show()

# Correlation Analysis

# Calculate the correlation matrix of the dataset
correlation_matrix = calories_data.corr()

# Plot the heatmap of the correlation matrix
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.1f', cmap='Blues', cbar=True, square=True, annot_kws={'size': 8})
plt.title('Correlation Heatmap')
plt.show()

# Data Preprocessing

# Convert the 'Gender' column from categorical text to numerical values (0 for male, 1 for female)
calories_data['Gender'] = calories_data['Gender'].map({'male': 0, 'female': 1})

# Show the updated DataFrame
print(calories_data.head())

# Separate the features (X) and target (Y)
X = calories_data.drop(columns=['User_ID', 'Calories'], axis=1)
Y = calories_data['Calories']

# Print the features and target
print(f"Features (X):\n{X.head()}")
print(f"Target (Y):\n{Y.head()}")

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Print the shapes of the training and testing sets
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# Model Training with XGBoost Regressor

# Initialize the XGBoost Regressor model
model = XGBRegressor()

# Train the model using the training data
model.fit(X_train, Y_train)

# Model Evaluation

# Predict the target variable (Calories) on the test data
test_predictions = model.predict(X_test)

# Print the predictions on the test data
print(f"Test Data Predictions:\n{test_predictions}")

# Calculate the Mean Absolute Error (MAE) between the actual and predicted values
mae = metrics.mean_absolute_error(Y_test, test_predictions)

# Print the Mean Absolute Error
print(f"Mean Absolute Error (MAE): {mae}")

