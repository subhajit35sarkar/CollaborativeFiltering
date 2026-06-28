import streamlit as st

from recommender import recommend_movies
from recommender import get_watched_movies

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬"
)

st.title("🎬 User-Based Movie Recommendation System")

st.write("Collaborative Filtering using Cosine Similarity")

# Dropdown instead of number input
user_id = st.selectbox(
    "Select User",
    list(range(1,611))
)

if st.button("Recommend Movies"):

    st.subheader("🍿 Previously Watched")

    watched = get_watched_movies(user_id)

    for movie in watched:
        st.write("⭐", movie)

    st.divider()

    st.subheader("🎯 Recommended Movies")

    recommendations = recommend_movies(user_id)

    if recommendations is not None:

        for movie in recommendations.index:

            st.write("🎬", movie)