import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import math

# Load the MovieLens 100K dataset
ratings_data = pd.read_csv("ratings.csv")

# Select a group of 3 random users
group_users = np.random.choice(ratings_data['userId'].unique(), size=3, replace=False)
group_ratings = ratings_data[ratings_data['userId'].isin(group_users)]

#(a) Function to calculate the Pearson correlation coefficient
def pearson_correlation(df_user1, df_user2):
    merged_ratings = df_user1.merge(df_user2, on="movieId", how="inner")
    if merged_ratings.empty:
        return math.nan
    
    ratings_user1 = merged_ratings['rating_x']
    ratings_user2 = merged_ratings['rating_y']
    mean_user1 = ratings_user1.mean()
    mean_user2 = ratings_user2.mean()
    
    num = np.sum((ratings_user1 - mean_user1) * (ratings_user2 - mean_user2))
    den = np.sqrt(np.sum((ratings_user1 - mean_user1) ** 2)) * np.sqrt(np.sum((ratings_user2 - mean_user2) ** 2))
    
    try:
        coef = num / den
    except (RuntimeWarning, ZeroDivisionError):
        return math.nan
    
    return coef

# Function to calculate the predicted rating for a movie
def predicted_rating(user_id, movie_id, ratings_data):
    df_userA = ratings_data[ratings_data['userId'] == user_id]
    userA_mean = df_userA['rating'].mean()
    users_for_film = ratings_data[ratings_data['movieId'] == movie_id].drop(['movieId', 'rating'], axis=1)
    num = 0
    den = 0
    
    for user in users_for_film['userId']:
        df_userB = ratings_data[ratings_data['userId'] == user]
        sim = pearson_correlation(df_userA, df_userB)
        if not math.isnan(sim):
            num += sim * (df_userB[df_userB['movieId'] == movie_id].iloc[0]['rating'] - df_userB['rating'].mean())
            den += sim

    try:
        div = num / den
        pred = userA_mean + div
        if math.isnan(pred):
            return None  # Return None if the predicted score is NaN
        return pred
    except (RuntimeWarning, ZeroDivisionError):
        return None  # Return None if there's an exception

# Function to calculate recommendations for a user
def calculate_recommendations(user_id, group_ratings):
    recommendations = []
    user_ratings = group_ratings[group_ratings['userId'] == user_id]
    all_movies = group_ratings['movieId'].unique()
    user_voted_movies = user_ratings['movieId'].unique()
    
    for movie_id in all_movies:
        if movie_id not in user_voted_movies:  # Check if the movie is not voted by the user
            predicted_score = predicted_rating(user_id, movie_id, group_ratings)
            if predicted_score is not None:
                recommendations.append((movie_id, predicted_score))  # Add only if the predicted score is not None
    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = recommendations[:10]

    return top_recommendations

# Function to aggregate recommendations using the average method
def aggregate_recommendations_average(recommendations):
    aggregated_scores = {}
    for user_rec in recommendations:
        for movie_id, predicted_score in user_rec:
            if movie_id not in aggregated_scores:
                aggregated_scores[movie_id] = [predicted_score]
            else:
                aggregated_scores[movie_id].append(predicted_score)

    averaged_recommendations = []
    for movie_id, scores in aggregated_scores.items():
        avg_score = np.mean([score for score in scores if not np.isnan(score)])
        if not np.isnan(avg_score):
            averaged_recommendations.append((movie_id, avg_score))

    return sorted(averaged_recommendations, key=lambda x: x[1], reverse=True)[:10]

# Function to aggregate recommendations using the least misery method
def aggregate_recommendations_least_misery(recommendations):
    least_misery_recommendations = {}
    for user_rec in recommendations:
        for movie_id, predicted_score in user_rec:
            if movie_id not in least_misery_recommendations:
                least_misery_recommendations[movie_id] = predicted_score
            else:
                least_misery_recommendations[movie_id] = min(least_misery_recommendations[movie_id], predicted_score)

    return sorted(least_misery_recommendations.items(), key=lambda x: x[1], reverse=True)[:10]

# Calculate recommendations for each user in the group
recommendations = {}
for user_id in group_users:
    recommendations[user_id] = calculate_recommendations(user_id, group_ratings)

# Print the users considered for recommendations
print("\nUsers considered for recommendations:")
for user_id in group_users:
    print(f"User ID: {user_id}")

# Aggregate recommendations using the average method
average_recommendations = aggregate_recommendations_average(list(recommendations.values()))

# Aggregate recommendations using the least misery method
least_misery_recommendations = aggregate_recommendations_least_misery(list(recommendations.values()))    

# Print the top-10 recommendations for the group using both aggregation methods
print("\nTop-10 recommendations using the average method for the group:")
for rank, (movie_id, predicted_score) in enumerate(average_recommendations, start=1):
    print(f"{rank}. Movie ID: {movie_id}, Predicted Score: {predicted_score}")

print("\nTop-10 recommendations using the least misery method for the group:")
for rank, (movie_id, predicted_score) in enumerate(least_misery_recommendations, start=1):
    print(f"{rank}. Movie ID: {movie_id}, Predicted Score: {predicted_score}")


#(b) Function to calculate disagreement among users
def calculate_disagreement(recommendations):
    movie_scores = {}
    for user_rec in recommendations:
        for movie_id, predicted_score in user_rec:
            if movie_id not in movie_scores:
                movie_scores[movie_id] = [predicted_score]
            else:
                movie_scores[movie_id].append(predicted_score)

    disagreement = {}
    for movie_id, scores in movie_scores.items():
        if len(scores) > 1:
            std_dev = np.std(scores)
            disagreement[movie_id] = std_dev

    return disagreement

# Function to aggregate recommendations using disagreement-aware method
def aggregate_recommendations_disagreement(recommendations, disagreement):
    aggregated_scores = {}
    for user_rec in recommendations:
        for movie_id, predicted_score in user_rec:
            if movie_id not in aggregated_scores:
                aggregated_scores[movie_id] = [predicted_score]
            else:
                aggregated_scores[movie_id].append(predicted_score)

    weighted_aggregated_recommendations = {}
    for movie_id, scores in aggregated_scores.items():
        avg_score = np.mean([score for score in scores if not np.isnan(score)])
        if not np.isnan(avg_score):
            weighted_aggregated_recommendations[movie_id] = avg_score * (1 - disagreement.get(movie_id, 0))

    return sorted(weighted_aggregated_recommendations.items(), key=lambda x: x[1], reverse=True)[:10]


# Calculate disagreement among users
disagreement = calculate_disagreement(list(recommendations.values()))

# Aggregate recommendations using the disagreement-aware method
disagreement_aware_recommendations = aggregate_recommendations_disagreement(list(recommendations.values()), disagreement)

# Print the top-10 recommendations for the group using the disagreement-aware method
print("\nTop-10 recommendations using the disagreement-aware method for the group:")
for rank, (movie_id, predicted_score) in enumerate(disagreement_aware_recommendations, start=1):
    print(f"{rank}. Movie ID: {movie_id}, Predicted Score: {predicted_score}")
