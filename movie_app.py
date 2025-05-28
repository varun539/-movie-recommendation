import streamlit as st
import pandas as pd
import ast
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load and preprocess movie data ---
movies = pd.read_csv("tmdb_5000_movies.csv")
movies = movies[['id', 'title', 'overview', 'genres', 'keywords']]
movies.dropna(inplace=True)

def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords']
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

# --- Vectorize and create similarity matrix ---
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

# --- TMDB poster fetch ---
def fetch_poster(movie_id):
    token = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyNTkwODI3OTE2MDczM2RlMjczNmFiMGY3OWNlYWNkZiIsIm5iZiI6MTc0ODQwMzgzMi4wNzMsInN1YiI6IjY4MzY4Njc4ZmE3YTc5YjgxMjgzODhhZSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.x-G1kVk6gQ6t2iApoaQpDaSgElFVWNr3E-BKuq-SSu4"
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        poster_path = res.json().get("poster_path")
        if poster_path:
            return "https://image.tmdb.org/t/p/w500" + poster_path
    except:
        return None

# --- Recommend movies ---
def recommend(movie_title):
    movie_title = movie_title.strip().lower()
    titles = movies['title'].str.lower().tolist()
    if movie_title not in titles:
        return []

    index = titles.index(movie_title)
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended = []
    for i in movie_list:
        movie_id = movies.iloc[i[0]].id
        title = movies.iloc[i[0]].title
        poster = fetch_poster(movie_id)
        recommended.append((title, poster))
    return recommended

# --- Streamlit UI ---
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")

selected_movie = st.text_input("Enter a movie name:")

if st.button("Recommend"):
    if selected_movie:
        recommendations = recommend(selected_movie)
        if not recommendations:
            st.error("❌ Movie not found or no recommendations available.")
        else:
            cols = st.columns(5)
            for i, (title, poster) in enumerate(recommendations):
                with cols[i]:
                    st.image(poster if poster else "https://via.placeholder.com/150", caption=title)
    else:
        st.warning("Please enter a movie name.")
