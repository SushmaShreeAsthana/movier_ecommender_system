from recommender import recommend_movies
from tmdb import get_movie_poster
import streamlit as st
from PIL import Image

# Custom CSS styling
st.markdown("""
    <style>
        body {
            background-color: #fdf6f0;
        }
        .main {
            background-color: #fdf6f0;
            color: #3a2e2a;
            font-family: 'Segoe UI', sans-serif;
        }
        h1, h3 {
            color: #3a2e2a;
        }
        .stTextInput > div > div > input {
            background-color: #fff7f1;
            color: #3a2e2a;
            padding: 0.5em 1em;
            border-radius: 10px;
            border: 1px solid #d3c0b0;
        }
        footer {
            visibility: hidden;
        }
        .custom-footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            padding: 10px;
            color: #7b5e4c;
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)

# Header section
st.markdown("<h1 style='font-size: 48px;'>🎬 NextWatch</h1>", unsafe_allow_html=True)
st.markdown("## Sip. Scroll. Stream.")
st.markdown("### <span style='color:#7b5e4c;'>Sometimes, all you need is a warm blanket and the right story.</span>", unsafe_allow_html=True)

# Input box
user_input = st.text_input("", placeholder="Search a movie you like...")


if user_input:
    st.markdown("## Recommendations for you:")
    recommended_movies = recommend_movies(user_input)

    cols = st.columns(len(recommended_movies))  # Create a horizontal row
    for idx, movie in enumerate(recommended_movies):
        with cols[idx]:
            poster_url = get_movie_poster(movie)
            if poster_url:
                st.image(poster_url, caption=movie, use_column_width=True)
            else:
                st.markdown(f"**{movie}**")


# Footer tagline
st.markdown("""
    <div class="custom-footer">
        Movie recommendations brewed to your taste – because your evenings deserve better.
    </div>
""", unsafe_allow_html=True)
