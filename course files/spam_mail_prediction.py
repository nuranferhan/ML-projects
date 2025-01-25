import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data Collection and Pre-Processing

# Load the raw email data from the CSV file into a DataFrame
raw_mail_data = pd.read_csv('mail_data.csv')
print(raw_mail_data)

# Replace any null values with an empty string
mail_data = raw_mail_data.fillna('')

# Display the first 5 rows of the DataFrame
print(mail_data.head())

# Check the shape of the DataFrame (number of rows and columns)
print(f"Data shape: {mail_data.shape}")

# Label Encoding

# Convert 'spam' to 0 and 'ham' to 1 in the 'Category' column
mail_data['Category'] = mail_data['Category'].map({'spam': 0, 'ham': 1})

# Display the category mappings
print("Category Mappings: 'spam' -> 0, 'ham' -> 1")

# Separate the dataset into text (X) and labels (Y)
X = mail_data['Message']
Y = mail_data['Category']

# Display the first few entries of X and Y
print(f"Messages (X):\n{X.head()}")
print(f"Categories (Y):\n{Y.head()}")

# Splitting the Data into Training and Test Sets

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Print the shapes of the training and test sets
print(f"X shape: {X.shape}")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Feature Extraction

# Initialize the TfidfVectorizer for text feature extraction
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

# Fit and transform the training data, and transform the test data
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Convert Y_train and Y_test to integers
Y_train = Y_train.astype(int)
Y_test = Y_test.astype(int)

# Display the transformed training features and their shape
print(f"Training features:\n{X_train_features}")
print(f"Shape of training features: {X_train_features.shape}")

# Model Training

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train_features, Y_train)

# Model Evaluation

# Predict the categories for the training data
train_predictions = model.predict(X_train_features)
train_accuracy = accuracy_score(Y_train, train_predictions)

# Print the accuracy on the training data
print(f"Accuracy on training data: {train_accuracy}")

# Predict the categories for the test data
test_predictions = model.predict(X_test_features)
test_accuracy = accuracy_score(Y_test, test_predictions)

# Print the accuracy on the test data
print(f"Accuracy on test data: {test_accuracy}")

# Building a Predictive System

# Input a new email for prediction
input_mail = ["I've been searching for the right words to thank you for this breather. I promise I won't take your help for granted and will fulfill my promise. You have been wonderful and a blessing at all times"]

# Transform the input email into feature vectors using the fitted vectorizer
input_features = vectorizer.transform(input_mail)

# Predict whether the input email is spam or ham
prediction = model.predict(input_features)

# Output the result
if prediction[0] == 1:
    print("Ham mail")
else:
    print("Spam mail")


