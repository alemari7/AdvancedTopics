import pandas as pd
import numpy as np

# (a) Load the MovieLens dataset
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')
links_df = pd.read_csv('links.csv')
tags_df = pd.read_csv('tags.csv')

# (b) Pearson correlation
def pearson_correlation(user1_ratings, user2_ratings):
    # Exclude NaN values
    common_movies = user1_ratings.dropna().index.intersection(user2_ratings.dropna().index)
    if len(common_movies) == 0:
        return 0  # No common movies, return 0 correlation
    else:
        user1_common_ratings = user1_ratings[common_movies]
        user2_common_ratings = user2_ratings[common_movies]
        # Compute mean ratings
        mean_user1 = user1_ratings.mean()
        mean_user2 = user2_ratings.mean()
        # Compute numerator and denominators
        numerator = ((user1_common_ratings - mean_user1) * (user2_common_ratings - mean_user2)).sum()
        denominator1 = np.sqrt(((user1_common_ratings - mean_user1)**2).sum())
        denominator2 = np.sqrt(((user2_common_ratings - mean_user2)**2).sum())
        # Handle division by zero
        if denominator1 == 0 or denominator2 == 0:
            return 0
        else:
            return numerator / (denominator1 * denominator2)

# (c) Prediction function for predicting movie scores
def predict_score(user_id, movie_id, similar_users, ratings_matrix):
    numerator, denominator = 0, 0
    mean_rating_user = ratings_matrix.loc[user_id].mean()  # Calculate the mean of user ratings
    
    for other_user_id, correlation in similar_users.items():
        if movie_id in ratings_matrix.loc[other_user_id].index:
            other_user_mean_rating = ratings_matrix.loc[other_user_id].mean()  # Mean of other user's ratings
            numerator += correlation * (ratings_matrix.loc[other_user_id, movie_id] - other_user_mean_rating)  # Add normalized rating
            denominator += abs(correlation)
    
    if denominator == 0:
        return 0  # If there are no similarities with other users, return 0

    predicted_score = mean_rating_user + (numerator / denominator)  # Add the mean to the normalized ratio
    return predicted_score



# (d) Recommender function to recommend movies for a user
def recommend_movies(user_id, similar_users, ratings_matrix):
    recommended_movies = []
    for movie_id in ratings_matrix.columns:
        # Check if the rating for the user and movie is NaN
        if ratings_matrix.loc[user_id, movie_id].isnull().any():
            # Predict the score for the movie
            predicted_score = predict_score(user_id, movie_id, similar_users, ratings_matrix)
            recommended_movies.append((movie_id, predicted_score))
    # Sort recommended movies based on predicted score (rating) in descending order
    recommended_movies = sorted(recommended_movies, key=lambda x: x[1], reverse=True)[:10]
    return recommended_movies


# Select a user (e.g., user_id = 1)
user_id = 1

# Calculate similarities between users using Pearson correlation
def find_similar_users(user_id, ratings_matrix):
    user_ratings = ratings_matrix.loc[user_id]
    similar_users = {}
    for other_user_id, other_user_ratings in ratings_matrix.iterrows():
        if other_user_id != user_id:
            correlation = pearson_correlation(user_ratings, other_user_ratings)
            similar_users[other_user_id] = correlation
    similar_users = dict(sorted(similar_users.items(), key=lambda x: x[1], reverse=True)[:10])
    return similar_users

# Find similar users for the selected user
similar_users = find_similar_users(user_id, ratings_df.set_index('userId'))

# Recommend movies for the selected user
recommended_movies = recommend_movies(user_id, similar_users, ratings_df.set_index('userId'))

# Display the results
print("10 most similar users for user", user_id, ":", similar_users)
print("10 most relevant movies recommended for user", user_id, ":", recommended_movies)
