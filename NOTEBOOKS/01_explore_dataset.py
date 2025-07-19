import pandas as pd
#import nltk
#nltk.download('punkt')

# Load the movies dataset
movies_df = pd.read_csv("data/tmdb_5000_movies.csv")

# Print the shape (rows, columns)
print("Shape of dataset:", movies_df.shape)

# Print the first 5 rows
print("\nFirst 5 entries:")
print(movies_df.head())

# Check for missing values
print("\nMissing values in each column:\n")
print(movies_df.isnull().sum())

# Check the data types
print("\nData types of each column:\n")
print(movies_df.dtypes)

# Optional: Check column names
print("\nColumn names:\n")
print(movies_df.columns)

# Drop columns that are not useful for recommendation
movies_df.drop(columns=['homepage', 'tagline'], inplace=True)

print("\nRemaining columns after dropping 'homepage' and 'tagline':")
print(movies_df.columns)

# Select only the features we care about
movies_df = movies_df[['id', 'title', 'genres', 'keywords', 'overview', 'original_language']]

print("\nUpdated dataset shape:", movies_df.shape)
print("\nSample entries:")
print(movies_df.head(3))

import ast  # For converting stringified lists into actual lists

def convert(obj):
    try:
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L
    except:
        return []

# Apply the convert function
movies_df['genres'] = movies_df['genres'].apply(convert)
movies_df['keywords'] = movies_df['keywords'].apply(convert)

# Show sample to check
print("\nSample after converting 'genres' and 'keywords':")
print(movies_df[['title', 'genres', 'keywords']].head(3))

# Convert list-based columns into space-separated strings
movies_df['genres'] = movies_df['genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
movies_df['keywords'] = movies_df['keywords'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
movies_df['overview'] = movies_df['overview'].apply(lambda x: x if isinstance(x, str) else '')

# Combine selected columns into a single string for each movie
movies_df['tags'] = movies_df['overview'] + ' ' + movies_df['genres'] + ' ' + movies_df['keywords']

# Convert all text to lowercase
movies_df['tags'] = movies_df['tags'].apply(lambda x: x.lower())

# Check sample
print("\nSample tags column:")
print(movies_df[['title', 'tags']].head(3))

from nltk.stem.porter import PorterStemmer
import re

ps = PorterStemmer()

def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove non-alphanumeric
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    # Tokenize and stem
    return ' '.join([ps.stem(word) for word in text.split()])

movies_df['tags'] = movies_df['tags'].apply(preprocess)

print("\nSample preprocessed tags:")
print(movies_df[['title', 'tags']].head(3))
#print(movies_df[['title', 'tags']].head(10))

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies_df['tags']).toarray()

# Load credits datasets
credits_df = pd.read_csv('data/tmdb_5000_credits.csv')

# Merge on 'id' (movies) and 'movie_id' (credits)
merged = movies_df.merge(credits_df, left_on='id', right_on='movie_id')

# Display first 5 rows
print("Merged dataset preview:")
print(merged.head())
print(merged.columns)


# Select only the necessary columns
movies = merged[['id', 'title_x', 'overview', 'genres', 'keywords', 'cast', 'crew']].copy()
movies.rename(columns={'title_x': 'title'}, inplace=True)

# Drop rows with any nulls
movies.dropna(inplace=True)

import ast

def convert(obj):
    try:
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L
    except:
        return []

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

#print(f"Remaining rows after dropping nulls: {movies.shape[0]}")
#print(movies.head(2))

def get_top_3_cast(obj):
    try:
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
            if len(L) == 3:
                break
        return L
    except:
        return []
    
movies['cast'] = movies['cast'].apply(get_top_3_cast)

def get_director(obj):
    try:
        L = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
        return L
    except:
        return []

movies['crew'] = movies['crew'].apply(get_director)
#print(movies[['title', 'cast', 'crew']].head(3))

def collapse(L):
    return " ".join(L)

movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)

# Convert overview to string and lowercase everything
movies['overview'] = movies['overview'].apply(lambda x: x if isinstance(x, str) else "")
movies['overview'] = movies['overview'].apply(lambda x: x.lower())

# Merge all into 'tags'
movies['tags'] = movies['overview'] + " " + movies['genres'] + " " + movies['keywords'] + " " + movies['cast'] + " " + movies['crew']
#print(movies[['title', 'tags']].head(3))

from nltk.stem.porter import PorterStemmer
import re

ps = PorterStemmer()

def clean_text(text):
    # Remove non-alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Tokenize
    text = text.lower().split()
    # Remove spaces in names and apply stemming
    text = [ps.stem(word) for word in text]
    return " ".join(text)

movies['tags'] = movies['tags'].apply(clean_text)
print(movies[['title', 'tags']].head(3))

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
print(vectors.shape)

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)

def recommend(movie):
    movie = movie.lower()
    try:
        index = movies[movies['title'].str.lower() == movie].index[0]
    except IndexError:
        print("Movie not found. Try a different title.")
        return

    distances = list(enumerate(similarity[index]))
    movies_list = sorted(distances, key=lambda x: x[1], reverse=True)[1:21]

    print(f"\nTop movies similar to '{movie.title()}':")
    for i in movies_list:
        print(movies.iloc[i[0]].title)
#recommend('Avatar')
import pickle
import gzip

# Save processed DataFrame (movies stays as is)
pickle.dump(movies, open('data/movies.pkl', 'wb'))

# Save similarity matrix as compressed file
with gzip.open('data/similarity.pkl.gz', 'wb') as f:
    pickle.dump(similarity, f)

import pickle
import gzip

movies = pickle.load(open('data/movies.pkl', 'rb'))

with gzip.open('data/similarity.pkl.gz', 'rb') as f:
    similarity = pickle.load(f)


print(movies.head())
print(similarity.shape)










