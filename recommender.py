import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("Data/movies.csv")
ratings = pd.read_csv("Data/ratings.csv")

df = ratings.merge(movies, on="movieId")

user_movie = df.pivot_table(
    index='userId',
    columns='title',
    values='rating'
)

user_movie_filled = user_movie.fillna(0)

user_similarity = cosine_similarity(user_movie_filled)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_filled.index,
    columns=user_movie_filled.index
)

def recommend_movies(user_id, k=20, n=10, min_ratings=3):

    if user_id not in user_movie.index:
        return None

    similar_users = (
        user_similarity_df.loc[user_id]
        .drop(user_id)
        .sort_values(ascending=False)
        .head(k)
    )

    top_users_ratings = user_movie.loc[similar_users.index].fillna(0)

    weighted_ratings = top_users_ratings.T.dot(similar_users)
    weighted_ratings = weighted_ratings / similar_users.sum()

    watched = user_movie.loc[user_id].dropna().index

    recommendations = weighted_ratings.drop(watched)
    recommendations = recommendations.dropna()

    rating_counts = top_users_ratings.notna().sum()

    filtered = recommendations[
        rating_counts[recommendations.index] >= min_ratings
    ]

    # If filtering removes everything,
    # return the unfiltered recommendations.
    if len(filtered) == 0:
        filtered = recommendations

    return filtered.sort_values(ascending=False).head(n)