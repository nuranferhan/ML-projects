import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data Loading and Preprocessing

# Read the dataset into a pandas DataFrame
sonar_data = pd.read_csv('sonar data.csv', header=None)

# Display the first few rows of the dataset
sonar_data.head()

# Show the shape of the dataset
sonar_data.shape

# Get statistical summary of the data
sonar_data.describe()

# Count the occurrences of each value in column 60
sonar_data[60].value_counts()

# M represents Mine and R represents Rock
sonar_data.groupby(60).mean()

# Splitting the dataset into features (X) and target labels (Y)
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

# Display the features and labels
print(X)
print(Y)

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# Display the shapes of the datasets
print(f"Shape of X: {X.shape}, X_train: {X_train.shape}, X_test: {X_test.shape}")

# Display the training features and labels
print(X_train)
print(Y_train)

# Model Training using Logistic Regression
model = LogisticRegression()

# Train the model with the training data
model.fit(X_train, Y_train)

# Model Evaluation

# Evaluate accuracy on the training data
train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_predictions)

print(f'Accuracy on training data: {train_accuracy}')

# Evaluate accuracy on the test data
test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_predictions)

print(f'Accuracy on test data: {test_accuracy}')

# Making Predictions with New Data

# Define the input data for prediction
input_data = (0.0307, 0.0523, 0.0653, 0.0521, 0.0611, 0.0577, 0.0665, 0.0664, 0.1460, 0.2792, 0.3877, 0.4992, 0.4981,
              0.4972, 0.5607, 0.7339, 0.8230, 0.9173, 0.9975, 0.9911, 0.8240, 0.6498, 0.5980, 0.4862, 0.3150, 0.1543,
              0.0989, 0.0284, 0.1008, 0.2636, 0.2694, 0.2930, 0.2925, 0.3998, 0.3660, 0.3172, 0.4609, 0.4374, 0.1820,
              0.3376, 0.6202, 0.4448, 0.1863, 0.1420, 0.0589, 0.0576, 0.0672, 0.0269, 0.0245, 0.0190, 0.0063, 0.0321,
              0.0189, 0.0137, 0.0277, 0.0152, 0.0052, 0.0121, 0.0124, 0.0055)

# Convert the input data into a numpy array
input_data_array = np.array(input_data)

# Reshape the array to match the model's expected input shape
input_data_reshaped = input_data_array.reshape(1, -1)

# Make a prediction
prediction = model.predict(input_data_reshaped)

# Output the prediction
if prediction[0] == 'R':
    print('The object is a Rock')
else:
    print('The object is a Mine')



