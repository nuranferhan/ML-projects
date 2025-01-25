import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# Data Collection & Initial Exploration

# Loading the data from CSV file into a Pandas DataFrame
big_mart_data = pd.read_csv('Train.csv')

# Displaying the first 5 rows of the DataFrame
print(big_mart_data.head())

# Checking the shape of the dataset (number of rows and columns)
print(f"Shape of the dataset: {big_mart_data.shape}")

# Displaying the dataset information (data types, non-null counts)
big_mart_data.info()

# Checking for missing values in the dataset
missing_values = big_mart_data.isnull().sum()
print(f"Missing Values:\n{missing_values}")

# Handling Missing Values

# Filling missing values in "Item_Weight" with its mean value
mean_item_weight = big_mart_data['Item_Weight'].mean()
big_mart_data['Item_Weight'].fillna(mean_item_weight, inplace=True)

# Filling missing values in "Outlet_Size" with the mode value for each Outlet_Type
mode_outlet_size = big_mart_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=lambda x: x.mode()[0])
print(f"Mode of Outlet Size for each Outlet Type:\n{mode_outlet_size}")

missing_outlet_size = big_mart_data['Outlet_Size'].isnull()
big_mart_data.loc[missing_outlet_size, 'Outlet_Size'] = big_mart_data.loc[missing_outlet_size, 'Outlet_Type'].apply(
    lambda x: mode_outlet_size[x]
)

# Verifying if there are any missing values left
print(f"Missing Values After Filling:\n{big_mart_data.isnull().sum()}")

# Data Analysis

# Displaying the statistical summary of the dataset
print(big_mart_data.describe())

# Plotting the distribution of Item_Weight
plt.figure(figsize=(6, 6))
sns.histplot(big_mart_data['Item_Weight'], kde=True, color='purple')
plt.title('Item Weight Distribution')
plt.show()

# Plotting the distribution of Item_Visibility
plt.figure(figsize=(6, 6))
sns.histplot(big_mart_data['Item_Visibility'], kde=True, color='orange')
plt.title('Item Visibility Distribution')
plt.show()

# Plotting the distribution of Item_MRP
plt.figure(figsize=(6, 6))
sns.histplot(big_mart_data['Item_MRP'], kde=True, color='green')
plt.title('Item MRP Distribution')
plt.show()

# Plotting the distribution of Item_Outlet_Sales
plt.figure(figsize=(6, 6))
sns.histplot(big_mart_data['Item_Outlet_Sales'], kde=True, color='red')
plt.title('Item Outlet Sales Distribution')
plt.show()

# Plotting the Outlet Establishment Year
plt.figure(figsize=(6, 6))
sns.countplot(x='Outlet_Establishment_Year', data=big_mart_data, palette='viridis')
plt.title('Outlet Establishment Year Distribution')
plt.show()

# Categorical Feature Analysis

# Plotting the distribution of Item_Fat_Content
plt.figure(figsize=(6, 6))
sns.countplot(x='Item_Fat_Content', data=big_mart_data, palette='Set2')
plt.title('Item Fat Content Distribution')
plt.show()

# Plotting the distribution of Item_Type
plt.figure(figsize=(30, 6))
sns.countplot(x='Item_Type', data=big_mart_data, palette='Paired')
plt.title('Item Type Distribution')
plt.show()

# Plotting the distribution of Outlet_Size
plt.figure(figsize=(6, 6))
sns.countplot(x='Outlet_Size', data=big_mart_data, palette='coolwarm')
plt.title('Outlet Size Distribution')
plt.show()

# Data Preprocessing

# Replacing 'Item_Fat_Content' with standardized values
big_mart_data['Item_Fat_Content'].replace(
    {'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}, inplace=True
)

# Label Encoding for Categorical Columns

encoder = LabelEncoder()

# Encoding categorical columns
categorical_columns = [
    'Item_Identifier', 'Item_Fat_Content', 'Item_Type',
    'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'
]

for column in categorical_columns:
    big_mart_data[column] = encoder.fit_transform(big_mart_data[column])

# Verifying the transformations
print(big_mart_data.head())

# Splitting Features and Target Variable

X = big_mart_data.drop(columns='Item_Outlet_Sales')
y = big_mart_data['Item_Outlet_Sales']

# Displaying the features and target variable
print(X.head())
print(y.head())

# Splitting the data into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Displaying the shape of the training and test sets
print(f"Training Set Shape: {X_train.shape}, Test Set Shape: {X_test.shape}")

# Machine Learning Model Training: XGBoost Regressor

# Initializing and training the XGBoost Regressor model
xgb_regressor = XGBRegressor()
xgb_regressor.fit(X_train, y_train)

# Model Evaluation

# Predicting on the training data
train_predictions = xgb_regressor.predict(X_train)

# Calculating the R-squared value for training data
train_r2 = metrics.r2_score(y_train, train_predictions)
print(f"Training R-squared: {train_r2}")

# Predicting on the test data
test_predictions = xgb_regressor.predict(X_test)

# Calculating the R-squared value for test data
test_r2 = metrics.r2_score(y_test, test_predictions)
print(f"Test R-squared: {test_r2}")


