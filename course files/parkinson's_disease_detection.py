import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Data Collection & Exploration

# Reading the dataset from a CSV file into a DataFrame
parkinsons_data = pd.read_csv('parkinsons.csv')

# Displaying the first 5 records in the dataset
print(parkinsons_data.head())

# Checking the shape of the DataFrame (number of rows and columns)
print(f"Shape of the dataset: {parkinsons_data.shape}")

# Displaying dataset information such as column types and non-null counts
parkinsons_data.info()

# Checking for missing values in the dataset
missing_values = parkinsons_data.isnull().sum()
print(f"Missing Values:\n{missing_values}")

# Displaying statistical summary of the data
print(parkinsons_data.describe())

# Distribution of the target variable
status_counts = parkinsons_data['status'].value_counts()
print(f"Status Distribution:\n{status_counts}")

# Target variable interpretation
print("""
1  --> Parkinson's Positive
0  --> Healthy
""")

# Grouping data by the target variable and calculating mean for each group
status_group = parkinsons_data.groupby('status').mean()
print(f"Grouped Data by Status:\n{status_group}")

# Data Preprocessing

# Separating features (X) and target (Y)
X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
Y = parkinsons_data['status']

# Displaying the feature matrix and target vector
print(f"Features (X):\n{X.head()}")
print(f"Target (Y):\n{Y.head()}")

# Splitting the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Displaying the shapes of the splits
print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Data Standardization

# Initializing the StandardScaler
scaler = StandardScaler()

# Fitting the scaler on the training data and transforming it
X_train = scaler.fit_transform(X_train)

# Transforming the test data using the same scaler
X_test = scaler.transform(X_test)

# Displaying the standardized training data
print(f"Standardized Training Data:\n{X_train[:5]}")

# Model Training: Support Vector Machine (SVM)

# Initializing the SVM classifier with a linear kernel
svm_model = SVC(kernel='linear')

# Training the model with the training data
svm_model.fit(X_train, Y_train)

# Model Evaluation: Accuracy Score

# Predicting on the training data
train_predictions = svm_model.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_predictions)

# Displaying the accuracy score for training data
print(f"Training Data Accuracy: {train_accuracy}")

# Predicting on the test data
test_predictions = svm_model.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_predictions)

# Displaying the accuracy score for test data
print(f"Test Data Accuracy: {test_accuracy}")

# Building a Predictive System

# Sample input data
input_data = (197.07600, 206.89600, 192.05500, 0.00289, 0.00001, 0.00166, 0.00168, 0.00498, 0.01098, 0.09700,
              0.00563, 0.00680, 0.00802, 0.01689, 0.00339, 26.77500, 0.422229, 0.741367, -7.348300, 0.177551,
              1.743867, 0.085569)

# Converting the input data into a NumPy array
input_array = np.asarray(input_data)

# Reshaping the array to match the model's input format
input_reshaped = input_array.reshape(1, -1)

# Standardizing the input data using the previously fitted scaler
input_standardized = scaler.transform(input_reshaped)

# Making a prediction with the trained model
prediction = svm_model.predict(input_standardized)

# Displaying the result
if prediction[0] == 0:
    print("The person does not have Parkinson's Disease.")
else:
    print("The person has Parkinson's Disease.")

