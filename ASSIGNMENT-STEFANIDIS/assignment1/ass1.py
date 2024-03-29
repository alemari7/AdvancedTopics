import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import math

#(a) load the data and choose the user and movie
USER_1 = 4
movie_id = 23
ratings_data = pd.read_csv("ratings.csv").drop(['timestamp'], axis=1)

#(b) defining the function to find similar users, using the pearson correlation
def find_similar_users(user_id, ratings_data):
    similarity = {}
    #dataframe of user ratings for the user
    user_ratings = ratings_data[ratings_data['userId'] == user_id]
    for other_user_id in ratings_data['userId'].unique():
        if other_user_id != user_id:
            #dataframe of user ratings for the other user
            other_user_ratings = ratings_data[ratings_data['userId'] == other_user_id]
            #dataframe of common movies between the user and the other user
            common_movies = pd.merge(user_ratings, other_user_ratings, on='movieId', how='inner')
            if len(common_movies) >= 2:  # Ensure enough common movies for correlation
                corr, _ = pearsonr(common_movies['rating_x'], common_movies['rating_y'])
                if not np.isnan(corr):  # Check if correlation is not NaN
                    similarity[other_user_id] = corr
    return dict(sorted(similarity.items(), key=lambda x: x[1], reverse=True)[:10])

similar_users = find_similar_users(USER_1, ratings_data)
print("\nTop 10 similar users for user", USER_1, "using Pearson Correlation are:", similar_users)

#(c)define the function to predict the rating
def predicted_rating(user_id, movie_id, similarity, ratings_data):
    user_ratings = ratings_data[ratings_data['userId'] == user_id]
    user_mean_rating = user_ratings['rating'].mean()

    #these variables will be used to calculate the weighted sum and the sum of weights
    weighted_sum = 0
    sum_of_weights = 0

    for similar_user_id, sim_value in similarity.items():
        similar_user_ratings = ratings_data[ratings_data['userId'] == similar_user_id]
        if movie_id in similar_user_ratings['movieId'].values:
            #if the user has rated the movie, then we can use it to predict the rating
            similar_user_rating = similar_user_ratings[similar_user_ratings['movieId'] == movie_id]['rating'].values[0]
            weighted_sum += sim_value * (similar_user_rating - similar_user_ratings['rating'].mean())
            sum_of_weights += abs(sim_value)
    #if there are no similar users, then we return the mean rating of the user
    if sum_of_weights == 0:
        return user_mean_rating
    else:
        #otherwise, we return the predicted rating
        predicted_score = user_mean_rating + weighted_sum / sum_of_weights
        return predicted_score

print("\nPredicted rating for user", USER_1, "and movie", movie_id, ":", predicted_rating(USER_1, movie_id, similar_users, ratings_data))

# Define the function to find the top 10 recommended movies
def top_recommended_movies(user_id, similarity, ratings_data):
    user_ratings = ratings_data[ratings_data['userId'] == user_id]
    unrated_movies = ratings_data[~ratings_data['movieId'].isin(user_ratings['movieId'])]

    recommendations = []
    #for each unrated movie, we predict the rating and add it to the recommendations list
    for movie_id in unrated_movies['movieId'].unique():
        predicted_score = predicted_rating(user_id, movie_id, similarity, ratings_data)
        recommendations.append((movie_id, predicted_score))
        #sort the recommendations by the predicted score
    recommendations.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = recommendations[:10]
    return top_recommendations

# Find the top 10 recommended movies
top_movies = top_recommended_movies(USER_1, similar_users, ratings_data)
print("\nTop 10 recommended films for user ", USER_1, "are:")
for movie_id, predicted_score in top_movies:
    print("Movie ID:", movie_id, "| Predicted Rating:", predicted_score)

# (e) define function to find similar users using Euclidean similarity
def find_similar_users_euclidean(USER_1, ratings_data):
    user_ratings = ratings_data[ratings_data['userId'] == USER_1]
    other_user_ratings = ratings_data[ratings_data['userId'] != USER_1]

    similarity = dict()  # Dictionary for similarity { key='userid' : value = similarity}

    while ratings_data.shape[0] > 0:
        other_user_id = float(ratings_data.iloc[0]['userId'])
        other_user_ratings = ratings_data[ratings_data['userId'] == other_user_id]

        common_films = pd.merge(user_ratings, other_user_ratings, how='inner', on=['movieId'])

        if not common_films.empty:
            # Computing Euclidean distance
            euclidean_distance = math.sqrt(((common_films['rating_x'] - common_films['rating_y']) ** 2).sum())
            similarity.update({other_user_id: 1 / (1 + euclidean_distance)})  # Using inverse of Euclidean distance for greater similarity with smaller distance

        ratings_data = ratings_data[ratings_data['userId'] != other_user_id]

    similarity = dict(sorted(similarity.items(), key=lambda x: x[1], reverse=True)[:10])
    return similarity

# finding similar users using Euclidean similarity
similar_users_euclidean = find_similar_users_euclidean(USER_1, ratings_data)
print("\nTop 10 similar users for user", USER_1, "using Euclidean similarity are:", similar_users_euclidean)