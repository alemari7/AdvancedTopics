import pandas as pd
import numpy as np
from collections import defaultdict

# Caricamento del dataset MovieLens 100K
ratings = pd.read_csv('ratings.csv')

# Genera un gruppo di 3 utenti casuali
group_users = np.random.choice(ratings['userId'].unique(), size=3, replace=False)

# Funzione per generare raccomandazioni personalizzate per ciascun utente
def generate_personalized_recommendations(user_id, group_ratings):
    session_length = 5  # Lunghezza della sessione per le raccomandazioni sequenziali
    user_ratings = group_ratings[group_ratings['userId'] == user_id]
    user_sessions = []
    for _, session in user_ratings.groupby('timestamp'):
        session_movies = session['movieId'].tolist()
        if len(session_movies) >= session_length:
            user_sessions.append(session_movies[:session_length])
    recommendations = []
    for session in user_sessions:
        last_movie = session[-1]
        candidate_movies = group_ratings[~group_ratings['movieId'].isin(session)]['movieId'].unique()
        scores = []
        for movie in candidate_movies:
            co_occurring_count = sum(1 for s in user_sessions if last_movie in s and movie in s)
            scores.append((movie, co_occurring_count))
        scores.sort(key=lambda x: x[1], reverse=True)
        recommendations.extend(scores[:10])
    return recommendations

# Funzione per aggregare le raccomandazioni del gruppo
def aggregate_recommendations(recommendations):
    aggregated_scores = defaultdict(int)
    for movie, score in recommendations:
        aggregated_scores[movie] += score
    aggregated_recommendations = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)
    return aggregated_recommendations[:10]

# Esegue 3 round di raccomandazioni
for round_num in range(1, 4):
    print(f"Round {round_num} Recommendations:")
    # Genera raccomandazioni personalizzate per ciascun utente nel gruppo
    user_recommendations = {}
    for user_id in group_users:
        user_recommendations[user_id] = generate_personalized_recommendations(user_id, ratings)
    # Aggrega le raccomandazioni del gruppo
    aggregated_recommendations = aggregate_recommendations([rec for recs in user_recommendations.values() for rec in recs])
    # Stampa le raccomandazioni del round
    for rank, (movie_id, score) in enumerate(aggregated_recommendations, start=1):
        print(f"{rank}. Movie ID: {movie_id}, Score: {score}")
    print()
