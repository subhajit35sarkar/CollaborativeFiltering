import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "user_movie.pkl"), "rb") as f:
    user_movie = pickle.load(f)

with open(os.path.join(BASE_DIR, "user_similarity.pkl"), "rb") as f:
    user_similarity_df = pickle.load(f)


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

    if len(filtered) == 0:
        filtered = recommendations

    return filtered.sort_values(ascending=False).head(n)



def get_watched_movies(user_id, n=10):

    if user_id not in user_movie.index:
        return []

    watched = user_movie.loc[user_id].dropna()

    watched = watched.sort_values(ascending=False)

    return watched.head(n).index.tolist()
