import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Data Collection

# Load the wine dataset into a Pandas DataFrame
wine_data = pd.read_csv('winequality-red.csv')

# Check the shape of the dataset (rows and columns)
print(wine_data.shape)

# Display the first 5 rows of the dataset
print(wine_data.head())

# Check for any missing values in the dataset
print(wine_data.isnull().sum())

# Data Analysis and Visualization

# Get the statistical summary of the dataset
print(wine_data.describe())

# Count the number of occurrences of each wine quality
sns.catplot(x='quality', data=wine_data, kind='count')

# Visualize the relationship between volatile acidity and quality
plt.figure(figsize=(5, 5))
sns.barplot(x='quality', y='volatile acidity', data=wine_data)

# Visualize the relationship between citric acid and quality
plt.figure(figsize=(5, 5))
sns.barplot(x='quality', y='citric acid', data=wine_data)

# Correlation Analysis

# Calculate the correlation matrix
correlation_matrix = wine_data.corr()

# Generate a heatmap to visualize the correlation between features
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.1f', cmap='Blues', cbar=True, square=True, annot_kws={'size': 8})

# Data Preprocessing

# Separate features (X) and target (Y)
X = wine_data.drop(columns='quality')

# Display the feature data
print(X)

# Label Binarization: Convert quality into binary labels
Y = wine_data['quality'].apply(lambda quality: 1 if quality >= 7 else 0)

# Display the target data
print(Y)

# Train-Test Split

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Display the shapes of the train and test sets
print(f"Y shape: {Y.shape}, Y_train shape: {Y_train.shape}, Y_test shape: {Y_test.shape}")

# Model Training

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier()

# Train the model on the training data
rf_model.fit(X_train, Y_train)

# Model Evaluation

# Make predictions on the test data
Y_test_predictions = rf_model.predict(X_test)

# Calculate the accuracy score of the model on the test data
accuracy = accuracy_score(Y_test, Y_test_predictions)

# Display the accuracy score
print(f'Accuracy: {accuracy}')

# Building a Predictive System

# Example input data (features of a wine sample)
input_sample = (7.5, 0.5, 0.36, 6.1, 0.071, 17.0, 102.0, 0.9978, 3.35, 0.8, 10.5)

# Convert the input data to a numpy array
input_array = np.array(input_sample)

# Reshape the input data to match the model's expected input format
input_array_reshaped = input_array.reshape(1, -1)

# Make a prediction based on the input data
wine_quality_prediction = rf_model.predict(input_array_reshaped)

# Display the prediction result
if wine_quality_prediction[0] == 1:
    print('Good Quality Wine')
else:
    print('Bad Quality Wine')








