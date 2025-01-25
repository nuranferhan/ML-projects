import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk

# Download the stopwords list
nltk.download('stopwords')

# Display the English stopwords
print(stopwords.words('english'))

# Data Preprocessing

# Load the dataset into a pandas DataFrame
dataset = pd.read_csv('train.csv')

# Check the shape of the dataset
print(dataset.shape)

# Display the first 5 rows of the dataset
print(dataset.head())

# Count the missing values in each column
print(dataset.isnull().sum())

# Fill missing values with an empty string
dataset.fillna('', inplace=True)

# Combine author name and news title into a single 'content' column
dataset['content'] = dataset['author'] + ' ' + dataset['title']

# Print the 'content' column
print(dataset['content'])

# Separate the features and labels
X_data = dataset.drop(columns='label', axis=1)
Y_data = dataset['label']

# Display the features and labels
print(X_data)
print(Y_data)

# Stemming: Reducing words to their root form
stemmer = PorterStemmer()


def apply_stemming(text):
    # Remove non-alphabetical characters
    text = re.sub('[^a-zA-Z]', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Split the text into words
    words = text.split()

    # Stem the words and remove stopwords
    words = [stemmer.stem(word) for word in words if word not in stopwords.words('english')]

    # Join the words back into a single string
    return ' '.join(words)


# Apply stemming to the 'content' column
dataset['content'] = dataset['content'].apply(apply_stemming)

# Print the processed 'content' column
print(dataset['content'])

# Separate the features (content) and labels (category)
X_data = dataset['content'].values
Y_data = dataset['label'].values

# Print the features and labels
print(X_data)
print(Y_data)

# Check the shape of the labels
print(Y_data.shape)

# Convert the text data into numerical format using TF-IDF
vectorizer = TfidfVectorizer()
X_data = vectorizer.fit_transform(X_data)

# Print the transformed feature matrix
print(X_data)

# Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, stratify=Y_data, random_state=2)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluate the model's performance

# Predict on the training data
train_predictions = model.predict(X_train)

# Calculate the accuracy of the model on the training data
train_accuracy = accuracy_score(Y_train, train_predictions)
print(f'Training Data Accuracy: {train_accuracy}')

# Predict on the test data
test_predictions = model.predict(X_test)

# Calculate the accuracy of the model on the test data
test_accuracy = accuracy_score(Y_test, test_predictions)
print(f'Test Data Accuracy: {test_accuracy}')

# Making a prediction on a new instance

# Select a sample from the test set
new_data = X_test[3]

# Predict the label of the new instance
prediction = model.predict([new_data])

# Print the prediction
print(prediction)

# Interpret the prediction
if prediction[0] == 0:
    print('The news is Real')
else:
    print('The news is Fake')

# Print the actual label for comparison
print(f'Actual Label: {Y_test[3]}')


