import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

file_path = '/kaggle/input/heart-failure-prediction/heart.csv'
df = pd.read_csv(file_path)

df_head = df.head()
df_info = df.info()
df_summary = df.describe()

if 'Unnamed: 32' in df.columns:
    df = df.drop(columns=['Unnamed: 32'])

df['HeartDisease'] = df['HeartDisease'].map({0: 'No', 1: 'Yes'})

missing_values = df.isnull().sum()

label_encoder = LabelEncoder()

df['Sex'] = label_encoder.fit_transform(df['Sex'])  # M -> 1, F -> 0
df['ExerciseAngina'] = label_encoder.fit_transform(df['ExerciseAngina'])  # Yes -> 1, No -> 0

df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG', 'ST_Slope'], drop_first=True)

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='HeartDisease', palette='viridis')
plt.title('Target Variable Distribution')
plt.xlabel('Heart Disease (0: No, 1: Yes)')
plt.ylabel('Count')
plt.show()

important_features = ['Cholesterol', 'MaxHR', 'Age', 'Oldpeak']
df[important_features].hist(bins=20, figsize=(12, 10), color='skyblue')
plt.suptitle('Distribution of Important Variables', fontsize=16)
plt.show()

df['cholesterol_maxHR_ratio'] = df['Cholesterol'] / df['MaxHR']
df['age_oldpeak_ratio'] = df['Age'] / df['Oldpeak']

df['cholesterol_maxHR_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
df['age_oldpeak_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)

df['cholesterol_maxHR_ratio'].fillna(df['cholesterol_maxHR_ratio'].median(), inplace=True)
df['age_oldpeak_ratio'].fillna(df['age_oldpeak_ratio'].median(), inplace=True)

print(df[['cholesterol_maxHR_ratio', 'age_oldpeak_ratio']].head())

X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_matrix)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

