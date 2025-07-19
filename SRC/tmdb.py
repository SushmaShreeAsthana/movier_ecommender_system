import requests

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

