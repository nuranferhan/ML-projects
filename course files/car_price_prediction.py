import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics

# Data Collection and Processing

# Read the car data from the CSV file into a Pandas DataFrame
car_data = pd.read_csv('car data.csv')

# Display the first 5 rows of the DataFrame
print(car_data.head())

# Check the shape of the dataset (number of rows and columns)
print(car_data.shape)

# Get a summary of the dataset's information
car_data.info()

# Check for missing values in the dataset
print(car_data.isnull().sum())

# Display the distribution of categorical data columns
print(car_data['Fuel_Type'].value_counts())
print(car_data['Seller_Type'].value_counts())
print(car_data['Transmission'].value_counts())

# Encoding the Categorical Data

# Map the 'Fuel_Type' column to numerical values
car_data['Fuel_Type'] = car_data['Fuel_Type'].map({'Petrol': 0, 'Diesel': 1, 'CNG': 2})

# Map the 'Seller_Type' column to numerical values
car_data['Seller_Type'] = car_data['Seller_Type'].map({'Dealer': 0, 'Individual': 1})

# Map the 'Transmission' column to numerical values
car_data['Transmission'] = car_data['Transmission'].map({'Manual': 0, 'Automatic': 1})

# Display the updated DataFrame
print(car_data.head())

# Splitting the data and target variable

# Separate the features (X) and the target (Y)
X = car_data.drop(columns=['Car_Name', 'Selling_Price'])
Y = car_data['Selling_Price']

# Display the features and target
print(X.head())
print(Y.head())

# Splitting the dataset into training and testing sets

# Split the data into training and testing sets (90% train, 10% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

# Model Training: Linear Regression

# Initialize the Linear Regression model
linear_model = LinearRegression()

# Train the model on the training data
linear_model.fit(X_train, Y_train)

# Model Evaluation: Linear Regression

# Make predictions on the training data
train_predictions = linear_model.predict(X_train)

# Calculate the R-squared error for the training data
train_r2_score = metrics.r2_score(Y_train, train_predictions)
print(f"Training R-squared Error: {train_r2_score}")

# Visualize the actual vs predicted prices for the training data
plt.scatter(Y_train, train_predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices (Training Data)")
plt.show()

# Make predictions on the test data
test_predictions = linear_model.predict(X_test)

# Calculate the R-squared error for the test data
test_r2_score = metrics.r2_score(Y_test, test_predictions)
print(f"Test R-squared Error: {test_r2_score}")

# Visualize the actual vs predicted prices for the test data
plt.scatter(Y_test, test_predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices (Test Data)")
plt.show()

# Model Training: Lasso Regression

# Initialize the Lasso Regression model
lasso_model = Lasso()

# Train the Lasso model on the training data
lasso_model.fit(X_train, Y_train)

# Model Evaluation: Lasso Regression

# Make predictions on the training data
lasso_train_predictions = lasso_model.predict(X_train)

# Calculate the R-squared error for the training data
lasso_train_r2_score = metrics.r2_score(Y_train, lasso_train_predictions)
print(f"Lasso Training R-squared Error: {lasso_train_r2_score}")

# Visualize the actual vs predicted prices for the training data (Lasso)
plt.scatter(Y_train, lasso_train_predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices (Training Data - Lasso)")
plt.show()

# Make predictions on the test data
lasso_test_predictions = lasso_model.predict(X_test)

# Calculate the R-squared error for the test data
lasso_test_r2_score = metrics.r2_score(Y_test, lasso_test_predictions)
print(f"Lasso Test R-squared Error: {lasso_test_r2_score}")

# Visualize the actual vs predicted prices for the test data (Lasso)
plt.scatter(Y_test, lasso_test_predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices (Test Data - Lasso)")
plt.show()


