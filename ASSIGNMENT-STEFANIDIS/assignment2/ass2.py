import pandas as pd
import numpy as np
import math

# (a) Load the MovieLens dataset
USER_1 = 7

# import csv file and drop unuseful colunmns
ratings = pd.read_csv("ratings.csv").drop(['timestamp'], axis=1)
user1_ratings = ratings[ratings['userId'] == USER_1] # dataframe that contains ratings of USER A
other_ratings = ratings[ratings['userId'] != USER_1] # dataframe that contains ratings of all other users

similarity = dict() # dict for similarity { key='userid' : value = similarity}

# rows are ordered by userId
while ratings.shape[0] > 0:

    # read the userId of the first row and select all the rows(ratings) for this user
    user2_id = float(ratings.iloc[0]['userId'])
    user2_ratings = ratings[ratings['userId'] == user2_id]

    # create a dataframe that contains only the ratings on the common movies between the two users
    commonFilms = pd.merge(user1_ratings, user2_ratings, how ='inner', on =['movieId'])

    if commonFilms.empty != True: # correlation on ratings for common movies

        sim = commonFilms['rating_x'].corr(commonFilms['rating_y'])
        if not math.isnan(sim): #drop the user when similarity is nan
            similarity.update({user2_id : sim})

    # remove from the dataframe the rows(ratings of user B) analyzed in this iteration
    ratings = ratings[ratings['userId'] != user2_id]

# sort similarity and take top 10
similarity = dict(sorted(similarity.items(), key=lambda x:x[1], reverse=True)[:10])

print("10 most similar users for user", USER_1, "are:", similarity)

# Prediction function for predicting movie scores
movie_id = 36

# Calculate the rating for the movie 
predicted_rating = user1_ratings['rating'].mean() + (sim * (other_ratings['rating'] - other_ratings.groupby('userId')['rating'].transform('mean'))).sum() / sim.sum()

print("Predicted rating for user", USER_1, "and movie", movie_id, ":", predicted_rating)

# Define aggregation methods
def average_method(recommended_movies):
    aggregated_list = {}
    for movie_id, predicted_score in recommended_movies:
        if movie_id not in aggregated_list:
            aggregated_list[movie_id] = predicted_score
        else:
            aggregated_list[movie_id] = (predicted_score.sum()).mean()
    return sorted(aggregated_list.items(), key=lambda x: x[1], reverse=True)[:10]

def least_misery_method(recommended_movies):
    aggregated_list = {}
    for movie_id, predicted_score in recommended_movies:
        if movie_id not in aggregated_list:
            aggregated_list[movie_id] = predicted_score
        else:
            aggregated_list[movie_id] = min(aggregated_list[movie_id], predicted_score)
    return sorted(aggregated_list.items(), key=lambda x: x[1], reverse=True)[:10]

# Generate recommendations for a group of 3 users
group_users = [1, 2, 3]
group_ratings = ratings[ratings['userId'].isin(group_users)]

# Calculate similarities between group users
similarities = {}
for user_id in group_users:
    user_ratings = group_ratings[group_ratings['userId'] == user_id].set_index('movieId')['rating']
    for other_user_id in group_users:
        if user_id != other_user_id:
            other_user_ratings = group_ratings[group_ratings['userId'] == other_user_id].set_index('movieId')['rating']
            correlation = sim(user_ratings, other_user_ratings)
            similarities[(user_id, other_user_id)] = correlation

# Calculate predictions for each user
all_recommendations = {}
for user_id in group_users:
    user_ratings = group_ratings[group_ratings['userId'] == user_id].set_index('movieId')['rating']
    similar_users = {other_user_id: correlation for (user1, user2), correlation in similarities.items() if user1 == user_id}
    recommendations = []
    for movie_id in group_ratings['movieId'].unique():
        if pd.isnull(group_ratings[(group_ratings['userId'] == user_id) & (group_ratings['movieId'] == movie_id)]['rating'].values):
            predicted_rating = predicted_rating(user_id, movie_id, similar_users, group_ratings.set_index('userId'))
            recommendations.append((movie_id, predicted_rating))
    all_recommendations[user_id] = recommendations

# Aggregate recommendations using average method
average_recommendations = []
for user_recommendations in all_recommendations.values():
    average_recommendations.extend(user_recommendations)
group_average_recommendations = average_method(average_recommendations)

# Aggregate recommendations using least misery method
least_misery_recommendations = []
for user_recommendations in all_recommendations.values():
    least_misery_recommendations.extend(user_recommendations)
group_least_misery_recommendations = least_misery_method(least_misery_recommendations)

# Print top-10 recommendations for each method
print("Top-10 recommendations using Average Method:")
for rank, (movie_id, predicted_score) in enumerate(group_average_recommendations, start=1):
    print(f"{rank}. Movie ID: {movie_id}, Predicted Score: {predicted_score}")

print("\nTop-10 recommendations using Least Misery Method:")
for rank, (movie_id, predicted_score) in enumerate(group_least_misery_recommendations, start=1):
    print(f"{rank}. Movie ID: {movie_id}, Predicted Score: {predicted_score}")
