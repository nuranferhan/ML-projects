import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Data Collection and Pre-Processing

# Read the movies dataset into a pandas DataFrame
movies_data = pd.read_csv('movies.csv')

# Display the first 5 rows of the dataset
print(movies_data.head())

# Get the shape of the dataset (number of rows and columns)
print(f"Dataset shape: {movies_data.shape}")

# Define the features to be used for movie recommendation
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
print(f"Selected features: {selected_features}")

# Replace null values in the selected features with an empty string
movies_data[selected_features] = movies_data[selected_features].fillna('')

# Combine all selected features into a single string for each movie
combined_features = movies_data[selected_features].apply(lambda x: ' '.join(x), axis=1)
print(combined_features.head())

# Convert the combined text data into feature vectors using TfidfVectorizer
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
print(f"Feature vectors shape: {feature_vectors.shape}")

# Cosine Similarity

# Calculate cosine similarity between the feature vectors
similarity_matrix = cosine_similarity(feature_vectors)
print(f"Similarity matrix shape: {similarity_matrix.shape}")

# Movie Recommendation Process

# Ask the user for their favorite movie name
movie_name = input('Enter your favorite movie name: ')

# Create a list of all movie titles from the dataset
all_movie_titles = movies_data['title'].tolist()
print(f"List of all movie titles: {all_movie_titles[:5]}")

# Find the closest match to the user's movie input
closest_match = difflib.get_close_matches(movie_name, all_movie_titles)
print(f"Closest match found: {closest_match}")

# Get the index of the closest matching movie
matched_movie_title = closest_match[0]
movie_index = movies_data[movies_data['title'] == matched_movie_title].index.values[0]
print(f"Index of the matched movie: {movie_index}")

# Get the similarity scores for the selected movie
similarity_scores = list(enumerate(similarity_matrix[movie_index]))
print(f"Similarity scores for the movie: {similarity_scores[:5]}")

# Sort the movies based on similarity scores in descending order
sorted_movies_by_similarity = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

# Display the top 30 similar movies
print('Movies suggested for you:')
for idx, (movie_idx, score) in enumerate(sorted_movies_by_similarity[:30], 1):
    movie_title = movies_data.iloc[movie_idx]['title']
    print(f"{idx}. {movie_title}")

# Repeat the movie recommendation system for a new movie input
movie_name = input('Enter your favorite movie name: ')
all_movie_titles = movies_data['title'].tolist()
closest_match = difflib.get_close_matches(movie_name, all_movie_titles)
matched_movie_title = closest_match[0]
movie_index = movies_data[movies_data['title'] == matched_movie_title].index.values[0]
similarity_scores = list(enumerate(similarity_matrix[movie_index]))
sorted_movies_by_similarity = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

# Display the top 30 recommended movies again
print('Movies suggested for you:')
for idx, (movie_idx, score) in enumerate(sorted_movies_by_similarity[:30], 1):
    movie_title = movies_data.iloc[movie_idx]['title']
    print(f"{idx}. {movie_title}")


