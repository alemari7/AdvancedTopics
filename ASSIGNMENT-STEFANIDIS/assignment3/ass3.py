import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import math

# Load the MovieLens 100K dataset
ratings_data = pd.read_csv("ratings.csv")

# Function to calculate the Pearson correlation coefficient
def pearson_correlation(df_user1, df_user2):
    # Merge ratings of two users on 'movieId'
    merged_ratings = df_user1.merge(df_user2, on="movieId", how="inner")
    if merged_ratings.empty:
        return math.nan
    
    # Extract ratings for each user
    ratings_user1 = merged_ratings['rating_x']
    ratings_user2 = merged_ratings['rating_y']
    
    # Calculate means of ratings
    mean_user1 = ratings_user1.mean()
    mean_user2 = ratings_user2.mean()
    
    # Calculate numerator and denominator for Pearson correlation
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
    users_for_film = ratings_data[ratings_data['movieId'] == movie_id]
    num = 0
    den = 0
    
    for _, userB_row in users_for_film.iterrows():
        userB = userB_row['userId']
        df_userB = ratings_data[ratings_data['userId'] == userB]
        sim = pearson_correlation(df_userA, df_userB)
        if not math.isnan(sim):
            rating_userB = userB_row['rating']
            num += sim * (rating_userB - df_userB['rating'].mean())
            den += sim

    try:
        div = num / den
        pred = userA_mean + div
        if math.isnan(pred):
            return None  # Return None if the predicted score is NaN
        return pred
    except (RuntimeWarning, ZeroDivisionError):
        return None  # Return None if there's an exception
    
# Define the function to calculate the recommendations for a user    
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

    return recommendations  # Return recommendations, even if empty

# Function to aggregate recommendations using the method of averaging
def average_aggregate_recommendations(recommendations):
    aggregated_scores = {}
    for user_rec in recommendations:
        for movie_id, predicted_score in user_rec:
            if movie_id not in aggregated_scores:
                aggregated_scores[movie_id] = [predicted_score]
            else:
                aggregated_scores[movie_id].append(predicted_score)

    average_aggregated_recommendations = []
    for movie_id, scores in aggregated_scores.items():
        avg_score = np.mean([score for score in scores if not np.isnan(score)])
        average_aggregated_recommendations.append((movie_id, avg_score))

    return sorted(average_aggregated_recommendations, key=lambda x: x[1], reverse=True)[:10]

# Function to aggregate recommendations using the method of least misery
def least_misery_aggregate_recommendations(recommendations):
    aggregated_scores = {}
    for user_rec in recommendations:
        for movie_id, predicted_score in user_rec:
            if movie_id not in aggregated_scores:
                aggregated_scores[movie_id] = [predicted_score]
            else:
                aggregated_scores[movie_id].append(predicted_score)

    least_misery_aggregated_recommendations = []
    for movie_id, scores in aggregated_scores.items():
        least_misery_score = min(scores)
        least_misery_aggregated_recommendations.append((movie_id, least_misery_score))

    return sorted(least_misery_aggregated_recommendations, key=lambda x: x[1], reverse=True)[:10]

# Function to aggregate recommendations using a hybrid method
def hybrid_aggregate_recommendations(recommendations, alpha):
    if alpha == 0:  # If alpha is zero, apply the averaging method
        return average_aggregate_recommendations(recommendations)
    elif alpha == 1:  # If alpha is one, apply the least misery method
        return least_misery_aggregate_recommendations(recommendations)
    else:  # Otherwise, use the hybrid method
        aggregated_scores = {}
    for user_rec in recommendations:
            for movie_id, predicted_score in user_rec:
                if movie_id not in aggregated_scores:
                    aggregated_scores[movie_id] = [predicted_score]
                else:
                    aggregated_scores[movie_id].append(predicted_score)

    hybrid_aggregated_recommendations = []
    for movie_id, scores in aggregated_scores.items():
            avg_score = np.mean([score for score in scores if not np.isnan(score)])
            least_misery_score = min(scores)
            hybrid_score = (avg_score * (1 - alpha)) + (least_misery_score * alpha)
            hybrid_aggregated_recommendations.append((movie_id, hybrid_score))

    return sorted(hybrid_aggregated_recommendations, key=lambda x: x[1], reverse=True)[:10]

# Function to generate sequential group recommendations using the hybrid method
def sequential_group_recommendations_hybrid(group_users, group_ratings, rounds=3, top_n=10):
    all_round_recommendations = []
    current_group_ratings = group_ratings.copy()  # Copy of initial input data
    
    for round_num in range(rounds):
        round_recommendations = []
        for user_id in group_users:
            user_recommendations = calculate_recommendations(user_id, current_group_ratings)
            round_recommendations.append(user_recommendations)  # Add user's recommendations to the list

        # Calculate alpha as the difference between max and min satisfaction values
        group_satisfaction_values = [user_recommendations[0][1] for user_recommendations in round_recommendations]
        alpha = max(group_satisfaction_values) - min(group_satisfaction_values)

        # Determine which aggregation method to use based on alpha
        if alpha == 0:  # If alpha is zero, apply the averaging method
            hybrid_round_recommendations = average_aggregate_recommendations(round_recommendations)
        elif alpha == 1:  # If alpha is one, apply the least misery method
            hybrid_round_recommendations = least_misery_aggregate_recommendations(round_recommendations)
        else:  # Otherwise, use the hybrid method
            hybrid_round_recommendations = hybrid_aggregate_recommendations(round_recommendations, alpha)

        all_round_recommendations.append(hybrid_round_recommendations)
        
        # Remove recommended movies from input data
        recommended_movies = [movie_id for movie_id, _ in hybrid_round_recommendations]
        current_group_ratings = current_group_ratings[~current_group_ratings['movieId'].isin(recommended_movies)]

    return all_round_recommendations

# Generate a group of 3 random users
group_users = np.random.choice(ratings_data['userId'].unique(), size=3, replace=False)
group_ratings = ratings_data[ratings_data['userId'].isin(group_users)]

# Calculate sequential group recommendations using hybrid method
sequential_recommendations_hybrid = sequential_group_recommendations_hybrid(group_users, group_ratings)

# Print the top-10 sequential group recommendations for each round using hybrid method
for round_num, round_recommendations in enumerate(sequential_recommendations_hybrid, start=1):
    print(f"\nRound {round_num} - Top-10 recommendations (Hybrid Method, 1st version):")
    if round_recommendations:
        for rank, (movie_id, predicted_score) in enumerate(round_recommendations, start=1):
            print(f"{rank}. Movie ID: {movie_id}, Predicted Score: {predicted_score}")
    else:
        print("No recommendations for this round.")