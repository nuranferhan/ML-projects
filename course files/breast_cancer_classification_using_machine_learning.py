import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data Collection & Processing

# Load the breast cancer dataset from sklearn
breast_cancer_data = sklearn.datasets.load_breast_cancer()

# Display the dataset details
print(breast_cancer_data)

# Convert the dataset into a pandas DataFrame
df = pd.DataFrame(breast_cancer_data.data, columns=breast_cancer_data.feature_names)

# Show the first 5 rows of the DataFrame
print(df.head())

# Add the target column (label) to the DataFrame
df['label'] = breast_cancer_data.target

# Show the last 5 rows of the DataFrame
print(df.tail())

# Get the shape of the dataset (number of rows and columns)
print(f"Dataset shape: {df.shape}")

# Get information about the dataset
print(df.info())

# Check for any missing values in the dataset
print(df.isnull().sum())

# Display statistical measures for the dataset
print(df.describe())

# Check the distribution of the target variable
print(df['label'].value_counts())

# Target label interpretation
# 1 --> Benign
# 0 --> Malignant

# Calculate the mean of each feature grouped by label
print(df.groupby('label').mean())

# Separating features and target variable
X = df.drop(columns='label')
y = df['label']

# Show the feature set and target variable
print(X.head())
print(y.head())

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Display the shapes of the training and testing sets
print(f"X shape: {X.shape}, X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# Model Training - Logistic Regression
logistic_model = LogisticRegression()

# Train the model using the training data
logistic_model.fit(X_train, y_train)

# Model Evaluation - Accuracy Score

# Evaluate accuracy on the training data
train_pred = logistic_model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)
print(f"Training accuracy: {train_accuracy}")

# Evaluate accuracy on the testing data
test_pred = logistic_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)
print(f"Testing accuracy: {test_accuracy}")

# Building a Predictive System

# Example input data for prediction
input_features = (13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259)

# Convert the input data to a numpy array
input_data_array = np.asarray(input_features)

# Reshape the array for a single prediction
reshaped_input = input_data_array.reshape(1, -1)

# Predict the label for the input data
prediction = logistic_model.predict(reshaped_input)

# Display the prediction result
print(prediction)

# Output the result based on the prediction
if prediction[0] == 0:
    print("The breast cancer is Malignant.")
else:
    print("The breast cancer is Benign.")


