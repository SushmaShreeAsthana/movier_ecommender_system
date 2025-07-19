# src/recommend.py
import pickle
import pandas as pd
import requests

# Load pickled data
movies = pickle.load(open('data/movies.pkl', 'rb'))
import gzip
import pickle

with gzip.open('data/similarity.pkl.gz', 'rb') as f:
    similarity = pickle.load(f)

#similarity = pickle.load(open('data/similarity.pkl.gz', 'rb'))

# OMDb API key
OMDB_API_KEY = "1a1e08ff"

def get_movie_poster(movie_title):
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={OMDB_API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    if data.get("Response") == "True":
        poster_url = data.get("Poster")
        if poster_url and poster_url != "N/A":
            return poster_url
    return None

def recommend_movies(movie):
    movie = movie.lower()
    movie_index = movies[movies['title'].str.lower() == movie].index
    if movie_index.empty:
        return []
    
    movie_index = movie_index[0]
    distances = sorted(
        list(enumerate(similarity[movie_index])),
        reverse=True,
        key=lambda x: x[1]
    )
    
    recommended_movie_names = []
    for i in distances[1:6]:  # top 5 recommendations
        title = movies.iloc[i[0]].title
        poster = get_movie_poster(title)
        recommended_movie_names.append((title, poster))
    
    return recommended_movie_names
