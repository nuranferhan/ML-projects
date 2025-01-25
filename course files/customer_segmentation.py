import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Data Collection and Exploration

# Load the dataset from a CSV file into a Pandas DataFrame
customer_data = pd.read_csv('Mall_Customers.csv')

# Display the first 5 rows of the DataFrame
print(customer_data.head())

# Get the shape of the dataset (rows and columns)
print(f"Shape of the dataset: {customer_data.shape}")

# Display dataset information (column types, non-null counts, etc.)
customer_data.info()

# Check for missing values in the dataset
missing_values = customer_data.isnull().sum()
print(f"Missing Values:\n{missing_values}")

# Selecting the 'Annual Income' and 'Spending Score' columns
X = customer_data.iloc[:, [3, 4]].values
print(f"Selected Data:\n{X}")

# Determining the optimal number of clusters using WCSS (Within-Cluster Sum of Squares)

# Initialize an empty list to store WCSS values
wcss = []

# Loop through different numbers of clusters (1 to 10)
for num_clusters in range(1, 11):
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
    kmeans.fit(X)

    # Append the inertia (WCSS) value to the list
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method to visualize the optimal number of clusters
sns.set()
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# From the Elbow graph, we determine the optimal number of clusters is 5

# Training the KMeans model with 5 clusters
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)

# Predict the cluster labels for each data point
Y = kmeans.fit_predict(X)
print(f"Cluster Labels:\n{Y}")

# Visualizing the clusters and their centroids

# Create a scatter plot for each cluster
plt.figure(figsize=(8, 8))

# Plot each cluster with a different color
plt.scatter(X[Y == 0, 0], X[Y == 0, 1], s=50, c='green', label='Cluster 1')
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y == 2, 0], X[Y == 2, 1], s=50, c='yellow', label='Cluster 3')
plt.scatter(X[Y == 3, 0], X[Y == 3, 1], s=50, c='violet', label='Cluster 4')
plt.scatter(X[Y == 4, 0], X[Y == 4, 1], s=50, c='blue', label='Cluster 5')

# Plot the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')

# Adding titles and labels to the plot
plt.title('Customer Segments Based on Income and Spending')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()


