import pandas as pd
import math

USER_1 = 24

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

print("10 most similar users for user1 are:", similarity)

# Prediction function for predicting movie scores
movie_id = 36

# Calculate the rating for the movie 
predicted_rating = user1_ratings['rating'].mean() + (sim * (other_ratings['rating'] - other_ratings.groupby('userId')['rating'].transform('mean'))).sum() / sim.sum()

print("Predicted rating for user", USER_1, "and movie", movie_id, ":", predicted_rating)


