import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Loading the file and inspecting the first few rows
file_path = '/kaggle/input/videogamesales/vgsales.csv'
df = pd.read_csv(file_path)

# Checking the data structure and the first rows
df_info = df.info()
df_head = df.head()

df_info, df_head

# Checking the status of missing values
missing_values = df.isnull().sum()

# Handling missing values:
# 1. Missing values in the 'Year' column: fill with -1.
# 2. Missing values in the 'Publisher' column: fill with 'Unknown'.
df['Year'].fillna(-1, inplace=True)
df['Publisher'].fillna('Unknown', inplace=True)

# Verifying that missing values have been cleaned
missing_values_after = df.isnull().sum()

missing_values, missing_values_after

# Distribution of Global_Sales
plt.figure(figsize=(10, 6))
sns.histplot(df['Global_Sales'], bins=30, kde=True, color='blue')
plt.title('Global Sales Distribution', fontsize=16)
plt.xlabel('Global Sales (in millions)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Top platforms by total global sales
platform_sales = df.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
platform_sales.plot(kind='bar', color='orange', alpha=0.8)
plt.title('Top 10 Platforms by Global Sales', fontsize=16)
plt.xlabel('Platform', fontsize=12)
plt.ylabel('Global Sales (in millions)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.show()

# Correlation between regional sales and Global_Sales
regional_sales = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
correlation = df[regional_sales].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='Blues', fmt=".2f")
plt.title('Correlation between Regional and Global Sales', fontsize=16)
plt.show()

# Impact of genres (Genre) on global sales
genre_sales = df.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
genre_sales.plot(kind='bar', color='green', alpha=0.8)
plt.title('Global Sales by Genre', fontsize=16)
plt.xlabel('Genre', fontsize=12)
plt.ylabel('Global Sales (in millions)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.show()

# Label Encoding for categorical variables
categorical_columns = ['Platform', 'Genre', 'Publisher']
label_encoders = {}

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Defining model input (X) and output (y)
X = df.drop(columns=['Rank', 'Name', 'Global_Sales'])  # Input features
y = df['Global_Sales']  # Target variable

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape

# Creating and training the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mse, mae, r2




