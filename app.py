import streamlit as st
from recommender import recommend_movies

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬"
)

st.title("🎬 Movie Recommendation System")

st.write("User-Based Collaborative Filtering")

user_id = st.number_input(
    "Enter User ID",
    min_value=1,
    max_value=610,
    value=1,
    step=1
)

if st.button("Recommend"):

    recommendations = recommend_movies(user_id)

    st.subheader("Recommended Movies")

    if recommendations is not None:

        for movie in recommendations.index:
            st.write("⭐", movie)