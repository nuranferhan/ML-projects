import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Data Collection & Initial Exploration

# Reading the CSV data into a Pandas DataFrame
insurance_data = pd.read_csv('insurance.csv')

# Displaying the first 5 rows of the DataFrame
print(insurance_data.head())

# Checking the number of rows and columns
print(f"Dataset Shape: {insurance_data.shape}")

# Getting information about the dataset (columns, data types, non-null counts)
insurance_data.info()

# Checking for missing values in the dataset
missing_values = insurance_data.isnull().sum()
print(f"Missing Values:\n{missing_values}")

# Statistical Summary of the dataset
print(insurance_data.describe())

# Data Visualization

# Plotting the distribution of age
plt.figure(figsize=(6, 6))
sns.histplot(insurance_data['age'], kde=True, color='blue')
plt.title('Age Distribution')
plt.show()

# Plotting the distribution of gender
plt.figure(figsize=(6, 6))
sns.countplot(x='sex', data=insurance_data, palette='Set2')
plt.title('Gender Distribution')
plt.show()

# Displaying the count of each gender
print(insurance_data['sex'].value_counts())

# Plotting the distribution of BMI
plt.figure(figsize=(6, 6))
sns.histplot(insurance_data['bmi'], kde=True, color='green')
plt.title('BMI Distribution')
plt.show()

# Plotting the distribution of children
plt.figure(figsize=(6, 6))
sns.countplot(x='children', data=insurance_data, palette='Set1')
plt.title('Children Distribution')
plt.show()

# Displaying the count of children
print(insurance_data['children'].value_counts())

# Plotting the distribution of smokers
plt.figure(figsize=(6, 6))
sns.countplot(x='smoker', data=insurance_data, palette='coolwarm')
plt.title('Smoker Distribution')
plt.show()

# Displaying the count of smokers
print(insurance_data['smoker'].value_counts())

# Plotting the distribution of region
plt.figure(figsize=(6, 6))
sns.countplot(x='region', data=insurance_data, palette='Pastel1')
plt.title('Region Distribution')
plt.show()

# Displaying the count of regions
print(insurance_data['region'].value_counts())

# Plotting the distribution of charges
plt.figure(figsize=(6, 6))
sns.histplot(insurance_data['charges'], kde=True, color='red')
plt.title('Charges Distribution')
plt.show()

# Data Preprocessing: Encoding Categorical Features

# Encoding the 'sex' column
insurance_data['sex'] = insurance_data['sex'].map({'male': 0, 'female': 1})

# Encoding the 'smoker' column
insurance_data['smoker'] = insurance_data['smoker'].map({'yes': 0, 'no': 1})

# Encoding the 'region' column
insurance_data['region'] = insurance_data['region'].map({
    'southeast': 0,
    'southwest': 1,
    'northeast': 2,
    'northwest': 3
})

# Splitting Features and Target Variable

X = insurance_data.drop(columns='charges')
y = insurance_data['charges']

# Displaying the features and target
print(X.head())
print(y.head())

# Splitting the data into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Displaying the shape of the splits
print(f"Feature Set Shape: {X.shape}, Training Set Shape: {X_train.shape}, Test Set Shape: {X_test.shape}")

# Model Training: Linear Regression

# Initializing the Linear Regression model
linear_model = LinearRegression()

# Training the model with the training data
linear_model.fit(X_train, y_train)

# Model Evaluation

# Predicting on the training data
train_predictions = linear_model.predict(X_train)

# Calculating the R-squared value for the training data
train_r2 = metrics.r2_score(y_train, train_predictions)
print(f"Training R-squared: {train_r2}")

# Predicting on the test data
test_predictions = linear_model.predict(X_test)

# Calculating the R-squared value for the test data
test_r2 = metrics.r2_score(y_test, test_predictions)
print(f"Test R-squared: {test_r2}")

# Building a Predictive System

# Example input data (age, sex, BMI, children, smoker, region)
input_data = (31, 1, 25.74, 0, 1, 0)

# Converting the input data into a NumPy array
input_array = np.array(input_data)

# Reshaping the array to fit the model's input format
input_reshaped = input_array.reshape(1, -1)

# Making the prediction
predicted_charge = linear_model.predict(input_reshaped)

# Displaying the prediction result
print(f"Predicted Insurance Charge: USD {predicted_charge[0]:.2f}")



