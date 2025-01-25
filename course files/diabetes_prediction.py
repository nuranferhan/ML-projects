import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the diabetes dataset into a pandas DataFrame
diabetes_data = pd.read_csv('diabetes.csv')

# Display the first 5 rows of the dataset
diabetes_data.head()

# Check the shape of the dataset (rows, columns)
diabetes_data.shape

# Get statistical summary of the dataset
diabetes_data.describe()

# Count the occurrences of each value in the 'Outcome' column
diabetes_data['Outcome'].value_counts()

# Explanation: 0 represents Non-Diabetic, 1 represents Diabetic

# Group by 'Outcome' and calculate the mean for each group
diabetes_data.groupby('Outcome').mean()

# Separate the features (X) and target variable (Y)
X = diabetes_data.drop(columns='Outcome', axis=1)
Y = diabetes_data['Outcome']

# Display the features and labels
print(X)
print(Y)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the feature data
scaler.fit(X)

# Transform the data to standardized values
standardized_X = scaler.transform(X)

# Display the standardized data
print(standardized_X)

# Update X with the standardized data
X = standardized_X
Y = diabetes_data['Outcome']

# Display the updated data
print(X)
print(Y)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Display the shapes of the datasets
print(X.shape, X_train.shape, X_test.shape)

# Initialize the SVM classifier with a linear kernel
svm_classifier = svm.SVC(kernel='linear')

# Train the classifier on the training data
svm_classifier.fit(X_train, Y_train)

# Predict the outcomes for the training data
train_predictions = svm_classifier.predict(X_train)

# Calculate and display the accuracy score for the training data
train_accuracy = accuracy_score(train_predictions, Y_train)
print('Training data accuracy score:', train_accuracy)

# Predict the outcomes for the test data
test_predictions = svm_classifier.predict(X_test)

# Calculate and display the accuracy score for the test data
test_accuracy = accuracy_score(test_predictions, Y_test)
print('Test data accuracy score:', test_accuracy)

# Example input data for prediction (patient's health parameters)
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

# Convert the input data into a numpy array
input_data_array = np.asarray(input_data)

# Reshape the data to match the expected input format for a single prediction
input_data_reshaped = input_data_array.reshape(1, -1)

# Standardize the input data using the previously fitted scaler
standardized_input = scaler.transform(input_data_reshaped)
print(standardized_input)

# Make the prediction using the trained model
prediction = svm_classifier.predict(standardized_input)
print(prediction)

# Interpret the prediction
if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')