import pandas as pd
import numpy as np
from scipy.stats import pearsonr

#load the MovieLens 100K dataset
ratings_data = pd.read_csv("ratings.csv")

#select a random group of 3 users
group_users = np.random.choice(ratings_data['userId'].unique(), size=3, replace=False)
group_ratings = ratings_data[ratings_data['userId'].isin(group_users)]

#define the function to calculate recommendations for a user
def calculate_recommendations(user_id, group_ratings):
    similarity = {}
    user_ratings = group_ratings[group_ratings['userId'] == user_id]
    for other_user_id in group_ratings['userId'].unique():
        if other_user_id != user_id:
            other_user_ratings = group_ratings[group_ratings['userId'] == other_user_id]
            common_movies = pd.merge(user_ratings, other_user_ratings, on='movieId', how='inner')
            if len(common_movies) >= 2:  #Ensure enough common movies for correlation
                corr, _ = pearsonr(common_movies['rating_x'], common_movies['rating_y'])
                if not np.isnan(corr):  #Check if correlation is not NaN
                    similarity[other_user_id] = corr

    recommendations = []
    for movie_id in group_ratings['movieId'].unique():
        predicted_score = predicted_rating(user_id, movie_id, similarity, group_ratings)
        recommendations.append((movie_id, predicted_score))
    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:10]


#define the function to calculate the predicted rating for a movie
def predicted_rating(user_id, movie_id, similarity, group_ratings):
    user_ratings = group_ratings[group_ratings['userId'] == user_id]
    user_mean_rating = user_ratings['rating'].mean()

    weighted_sum = 0
    sum_of_weights = 0

    for similar_user_id, sim_value in similarity.items():
        similar_user_ratings = group_ratings[group_ratings['userId'] == similar_user_id]
        if movie_id in similar_user_ratings['movieId'].values:
            similar_user_rating = similar_user_ratings[similar_user_ratings['movieId'] == movie_id]['rating'].values[0]
            weighted_sum += sim_value * (similar_user_rating - similar_user_ratings['rating'].mean())
            sum_of_weights += abs(sim_value)
    
    if sum_of_weights == 0:
        return user_mean_rating
    else:
        predicted_score = user_mean_rating + weighted_sum / sum_of_weights
        return predicted_score

#calculate recommendations for each user in the group
for user_id in group_users:
    user_recommendations = calculate_recommendations(user_id, group_ratings)
    print("\nRecommendations for user", user_id, ":")
    for rank, (movie_id, predicted_score) in enumerate(user_recommendations, start=1):
        print(f"{rank}. Movie ID: {movie_id}, Predicted Score: {predicted_score}")

#define the function to aggregate recommendations using the average method
def aggregate_recommendations_average(recommendations):
    aggregated_scores = {}
    for user_rec in recommendations.values():
        for movie_id, predicted_score in user_rec:
            if movie_id not in aggregated_scores:
                aggregated_scores[movie_id] = [predicted_score]
            else:
                aggregated_scores[movie_id].append(predicted_score)

    averaged_recommendations = []
    for movie_id, scores in aggregated_scores.items():
        avg_score = np.mean(scores)
        averaged_recommendations.append((movie_id, avg_score))

    return sorted(averaged_recommendations, key=lambda x: x[1], reverse=True)[:10]

#define the function to aggregate recommendations using the least misery method
def aggregate_recommendations_least_misery(recommendations):
    least_misery_recommendations = {}
    for user_rec in recommendations.values():
        for movie_id, predicted_score in user_rec:
            if movie_id not in least_misery_recommendations:
                least_misery_recommendations[movie_id] = predicted_score
            else:
                least_misery_recommendations[movie_id] = min(least_misery_recommendations[movie_id], predicted_score)

    return sorted(least_misery_recommendations.items(), key=lambda x: x[1], reverse=True)[:10]

#calculate recommendations for each user in the group
recommendations = {}
for user_id in group_users:
    recommendations[user_id] = calculate_recommendations(user_id, group_ratings)

#aggregate recommendations using the average method
average_recommendations = aggregate_recommendations_average(recommendations)

#aggregate recommendations using the least misery method
least_misery_recommendations = aggregate_recommendations_least_misery(recommendations)

#print the top-10 recommendations for the group
print("\nTop-10 recommendations using the average method for the group:")
for rank, (movie_id, predicted_score) in enumerate(average_recommendations, start=1):
    print(f"{rank}. Movie ID: {movie_id}, Predicted Score: {predicted_score}")

print("\nTop-10 recommendations using the least misery method for the group:")
for rank, (movie_id, predicted_score) in enumerate(least_misery_recommendations, start=1):
    print(f"{rank}. Movie ID: {movie_id}, Predicted Score: {predicted_score}")


#(b)define the function to calculate the disagreement between two users
def calculate_disagreement(user1_ratings, user2_ratings):
    #merge common movies between the two users
    common_movies = pd.merge(user1_ratings, user2_ratings, on='movieId', how='inner')
    
    #check if there are common movies to compute disagreement
    if not common_movies.empty:
        #calculate squared difference of ratings for common movies
        squared_diff = (common_movies['rating_x'].values - common_movies['rating_y'].values) ** 2
        #calculate the root mean square difference
        rmsd = np.sqrt(np.mean(squared_diff))
        return rmsd
    else:
        return None



#calculate Disagreements
disagreements = {}
for user1_id in group_users:
    for user2_id in group_users:
        if user1_id != user2_id:
            user1_ratings = group_ratings[group_ratings['userId'] == user1_id]
            user2_ratings = group_ratings[group_ratings['userId'] == user2_id]
            disagreement = calculate_disagreement(user1_ratings, user2_ratings)
            if disagreement is not None:
                disagreements[(user1_id, user2_id)] = disagreement

#define the function to calculate weighted recommendations for a group
def calculate_weighted_recommendations(group_users, group_ratings, disagreements):
    weighted_recommendations = {}
    for user_id in group_users:
        user_recommendations = {}
        for movie_id in group_ratings['movieId'].unique():
            predicted_score = predicted_rating(user_id, movie_id, disagreements, group_ratings)
            user_recommendations[movie_id] = predicted_score
        weighted_recommendations[user_id] = user_recommendations
    return weighted_recommendations

#define the function to aggregate weighted recommendations using average method
def aggregate_weighted_recommendations(weighted_recommendations):
    aggregated_scores = {}
    for user_rec in weighted_recommendations.values():
        for movie_id, predicted_score in user_rec.items():
            if movie_id not in aggregated_scores:
                aggregated_scores[movie_id] = [predicted_score]
            else:
                aggregated_scores[movie_id].append(predicted_score)
    
    averaged_recommendations = []
    for movie_id, scores in aggregated_scores.items():
        weighted_avg_score = np.mean(scores)
        averaged_recommendations.append((movie_id, weighted_avg_score))
    
    return sorted(averaged_recommendations, key=lambda x: x[1], reverse=True)[:10]

#calculate weighted recommendations for the group
weighted_recommendations = calculate_weighted_recommendations(group_users, group_ratings, disagreements)

#aggregate weighted recommendations using the average method
weighted_average_recommendations = aggregate_weighted_recommendations(weighted_recommendations)

#print the top-10 weighted average recommendations for the group
print("\nTop-10 recommendations for the group using weighted average method:")
for rank, (movie_id, weighted_avg_score) in enumerate(weighted_average_recommendations, start=1):
    print(f"{rank}. Movie ID: {movie_id}, Weighted Average Score: {weighted_avg_score}")



