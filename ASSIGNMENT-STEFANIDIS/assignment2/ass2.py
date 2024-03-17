import pandas as pd
import math
import numpy as np

#load the data and choose the user and movie
USER_1 = 76
movie_id = 77
ratings_data = pd.read_csv("ratings.csv").drop(['timestamp'], axis=1)

#defining the function to find similar users, using the pearson correlation
def find_similar_users(USER_1, ratings_data):
    # dataframe that contains ratings of the specified user
    user_ratings = ratings_data[ratings_data['userId'] == USER_1]
    # dataframe that contains ratings of all other users
    other_user_ratings = ratings_data[ratings_data['userId'] != USER_1]

    similarity = dict()  # dict for similarity { key='userid' : value = similarity}

    # rows are ordered by userId
    while ratings_data.shape[0] > 0:
        #select the first row and all the rows for this user
        other_user_id = float(ratings_data.iloc[0]['userId'])
        #extract the ratings of the other user and put into a dataframe
        other_user_ratings = ratings_data[ratings_data['userId'] == other_user_id]

        # create a dataframe that contains only the ratings on the common movies between the two users
        common_films = pd.merge(user_ratings, other_user_ratings, how='inner', on=['movieId'])

        if not common_films.empty:  # correlation on ratings for common movies
            sim = common_films['rating_x'].corr(common_films['rating_y'])
            if not math.isnan(sim):  # drop the user when similarity is nan
                similarity.update({other_user_id: sim})

        # remove from the dataframe the rows analyzed in this iteration
        ratings_data = ratings_data[ratings_data['userId'] != other_user_id]

    # sort similarity and take top 10 similar users
    similarity = dict(sorted(similarity.items(), key=lambda x: x[1], reverse=True)[:10])
    return similarity

similar_users = find_similar_users(USER_1, ratings_data)
print("Top 10 similar users for user", USER_1, "using Pearson Correlation are:", similar_users)

#defining the function to predict the rating
def predicted_rating(user_id, movie_id, similarity, ratings_data):
    user_ratings = ratings_data[ratings_data['userId'] == user_id]
    user_mean_rating = user_ratings['rating'].mean()

    #these variables will be used to calculate the weighted sum and the sum of weights
    weighted_sum = 0
    sum_of_weights = 0

    for similar_user_id, sim_value in similarity.items():
        similar_user_ratings = ratings_data[ratings_data['userId'] == similar_user_id]
        if movie_id in similar_user_ratings['movieId'].values:
            #if the user has rated the movie, then we can use the similarity to calculate the weighted sum
            similar_user_rating = similar_user_ratings[similar_user_ratings['movieId'] == movie_id]['rating'].values[0]
            weighted_sum += sim_value * (similar_user_rating - similar_user_ratings['rating'].mean())
            sum_of_weights += abs(sim_value)
    #if there are no similar users that have rated the movie, then we return the mean rating of the user
    if sum_of_weights == 0:
        return user_mean_rating
    else:
        #otherwise we return the predicted rating
        predicted_score = user_mean_rating + weighted_sum / sum_of_weights
        return predicted_score

print("Predicted rating for user", USER_1, "and movie", movie_id, ":", predicted_rating(USER_1, movie_id, similar_users, ratings_data))

#define the function to find the top 10 recommended movies
def top_recommended_movies(user_id, similarity, ratings_data):
    user_ratings = ratings_data[ratings_data['userId'] == user_id]
    unrated_movies = ratings_data[~ratings_data['movieId'].isin(user_ratings['movieId'])]

    recommendations = []
    #for each movie that the user has not rated, we calculate the predicted rating
    for movie_id in unrated_movies['movieId'].unique():
        predicted_score = predicted_rating(user_id, movie_id, similarity, ratings_data)
        recommendations.append((movie_id, predicted_score))
    #sort the recommendations by predicted rating
    recommendations.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = recommendations[:10]
    return top_recommendations

#find the top 10 recommended movies
top_movies = top_recommended_movies(USER_1, similar_users, ratings_data)
print("Top 10 recommended films for user ", USER_1, "are:")
for movie_id, predicted_score in top_movies:
    print("Movie ID:", movie_id, "| Predicted Rating:", predicted_score)

#define function to find similar users using Euclidean similarity
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
print("Top 10 similar users for user", USER_1, "using Euclidean similarity are:", similar_users_euclidean)


# (a)define aggregation methods
def average_method(top_movies):
    aggregated_scores = {}
    for movie_id, predicted_score in top_movies:
        if movie_id not in aggregated_scores:
            aggregated_scores[movie_id] = [predicted_score]
        else:
            aggregated_scores[movie_id].append(predicted_score)

    averaged_recommendations = []
    for movie_id, scores in aggregated_scores.items():
        avg_score = np.mean(scores)
        averaged_recommendations.append((movie_id, avg_score))

    return sorted(averaged_recommendations, key=lambda x: x[1], reverse=True)[:10]

#defining the least misery method
def least_misery_method(top_movies):
    aggregated_scores = {}
    for movie_id, predicted_score in top_movies:
        if movie_id not in aggregated_scores:
            aggregated_scores[movie_id] = [predicted_score]
        else:
            aggregated_scores[movie_id].append(predicted_score)

    least_misery_recommendations = []
    for movie_id, scores in aggregated_scores.items():
        min_score = min(scores)
        least_misery_recommendations.append((movie_id, min_score))

    return sorted(least_misery_recommendations, key=lambda x: x[1], reverse=True)[:10]

# Generate recommendations for a group of 3 users
group_users = [1, 2, 3]
group_ratings = ratings_data[ratings_data['userId'].isin(group_users)]

# Calculate similarities between group users
similarities = {}
for user_id in group_users:
    user_ratings = group_ratings[group_ratings['userId'] == user_id].set_index('movieId')['rating']
    for other_user_id in group_users:
        if user_id != other_user_id:
            other_user_ratings = group_ratings[group_ratings['userId'] == other_user_id].set_index('movieId')['rating']
            correlation = find_similar_users(USER_1, ratings_data)
            similarities[(user_id, other_user_id)] = correlation

# Calculate predictions for each user
all_recommendations = {}
for user_id in group_users:
    user_ratings = group_ratings[group_ratings['userId'] == user_id].set_index('movieId')['rating']
    similar_users = [(other_user_id, correlation) for (user1, user2), correlation in similarities.items() if user1 == user_id]
    recommendations = []
    for movie_id in group_ratings['movieId'].unique():
        if (group_ratings[(group_ratings['userId'] == user_id) & (group_ratings['movieId'] == movie_id)]['rating'].values.size > 0):
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
