import streamlit as st
import pandas as pd
import pickle
import requests
from dotenv import load_dotenv
import os
from  recommenderSystem import HybridRecommender, ratings, ContentBasedCF, generate_user_item_matrices, add_ratings, reload_and_retrain_model, update_model, get_movies_by_genre
import re

# --- TMDb Setup ---
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")  

def get_movie_poster(title, api_key):
    search_url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": api_key, "query": title}
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        data = response.json()

        # Check if 'results' exists and is not empty
        if 'results' in data and data['results']:
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        print(f"Error fetching poster for '{title}': {e}")
    
    # Fallback image if anything fails
    return "https://via.placeholder.com/300x450?text=No+Image"\
        
def clean_title(title):
      return re.sub(r"\s*\(.*?\)\s*", "", title).strip()

# --- Load Data ---
@st.cache_data #make it so the data wont change
def load_movies():
    return pd.read_csv("datasets/movies.csv")


def load_ratings():
    return pd.read_csv("datasets/ratings.csv")


def load_model():
    with open("model/recommender.pkl", "rb") as f:
        return pickle.load(f)


# --- Main App ---

st.set_page_config(page_title="StreamSage", layout="wide")

#set session_state
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "is_new_user" not in st.session_state:
    st.session_state.is_new_user = False
if "temp_recommendations" not in st.session_state:
    st.session_state.temp_recommendations = []

existing_user_ids = sorted(ratings['userId'].unique())
movies_df = load_movies()
model = load_model()


if st.session_state.user_id is None:
    st.subheader("Before we start, please choose your userID")
    option = st.radio("Choose an option:", ["Select existing user", "Create new user"])
    
    if option == "Select existing user":
        placeholder = "Select User...."
        selected_user_id = st.selectbox("Select your userID", options =[placeholder] + existing_user_ids)
    
        if selected_user_id != placeholder:
            st.session_state.user_id = int(selected_user_id)
            st.success(f"Welcome back, User {selected_user_id}!")
            st.rerun()

    elif option == "Create new user":
        new_user_id = st.number_input("Enter new userID", min_value=1, step=1)
        if st.button("Create new user"):
            if new_user_id in existing_user_ids:
                st.warning(f"UserID {new_user_id} already exists.")
            else:
                st.session_state.user_id = int(new_user_id)
                st.session_state.is_new_user = True  # Cold-start flag
                st.success(f"UserID {new_user_id} created!")
                st.rerun()
    
else:
    user_id = st.session_state.user_id
    if st.session_state.is_new_user:
        st.title("🎬 Movie Recommendation System")
        st.subheader(f"Hello user {user_id}")
        st.write("Select a movie to get personalized recommendations.")
        movie_titles = movies_df['title'].tolist()
        selected_movie = st.selectbox("Choose a movie", movie_titles, key="movie_selectbox")
        
        print(selected_movie)
    
        if st.button("Get Recommendations", key="get_recs"):
            with st.spinner("Generating recommendations..."):
                recommended_ids = model.hybrid_recommendation(user_id,movies_df,selected_movie, 5)
                st.session_state.temp_recommendations = recommended_ids
                print(f" test test {st.session_state.temp_recommendations}")
                st.subheader("🎥 Recommended for you:")
                cols = st.columns(5)
                for i, (movie_id,score) in enumerate(recommended_ids):
                    title_row = movies_df[movies_df['movieId'] == movie_id]
                    movie_title = title_row['title'].iloc[0] if not title_row.empty else "Unknown"
                    with cols[i]:
                        cleaned_title = clean_title(movie_title)
                        poster_url = get_movie_poster(cleaned_title, TMDB_API_KEY)
                        st.image(poster_url, use_container_width=True)
                        st.caption(f"{movie_title}")

        if st.session_state.temp_recommendations:
            st.subheader("Thank you for chosing your movie:")
            st.write("Please rate some of the movies given(1-5):")
            ratings_input = {}
            
            for movie_id,_ in st.session_state.temp_recommendations[:5]:
                title = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
                rating = st.slider(f"{title}", 1, 5, 3, key=f"rate_{movie_id}")
                ratings_input[movie_id] = rating
                
            if st.button("Submit Ratings"):
                st.session_state.user_ratings = ratings_input
                add_ratings(user_id, st.session_state.user_ratings)
                reload_and_retrain_model(user_id)
                
                with open("model/recommenderNew.pkl", "rb") as f:
                    modelNew = pickle.load(f)
                    
                st.subheader("New Recommendations")
                with st.spinner("Generating recommendations..."):   
                    recommended_ids = modelNew.hybrid_recommendation(user_id, movies_df, None, 5)
                  
                    cols = st.columns(5)
                    for i, (movie_id, score) in enumerate(recommended_ids):
                        title_row = movies_df[movies_df['movieId'] == movie_id]
                        movie_title = title_row['title'].iloc[0] if not title_row.empty else "Unknown"
                        

                        with cols[i]:
                            cleaned_title = clean_title(movie_title)
                            poster_url = get_movie_poster(cleaned_title, TMDB_API_KEY)
                            st.image(poster_url, use_container_width=True)
                            st.caption(f"{movie_title}\n⭐ Ratings: {score:.2f}")
        
                update_model()
               
              
        if st.button("Go Back to Main Menu"):
            st.session_state.user_id = None
            st.session_state.is_new_user = False
            st.rerun()
    else:
        st.title("🎬 Movie Recommendation System")
        st.subheader(f"Welcome back user {user_id}, here are some recommendations")
        st.subheader("Your Top 5 Rated Movies:")
        user_ratings = ratings[ratings['userId'] == user_id]
        top_rated = user_ratings.sort_values(by='rating', ascending=False).head(5)
        cols_top = st.columns(5)
        
        for i, movie_id in enumerate(top_rated['movieId']):
            title_row = movies_df[movies_df['movieId'] == movie_id]
            movie_title = title_row['title'].iloc[0] if not title_row.empty else "Unknown"
            
            cleaned_title = clean_title(movie_title)
            poster_url = get_movie_poster(cleaned_title, TMDB_API_KEY)
            
            with cols_top[i]:
                st.image(poster_url, use_container_width=True)
                st.caption(f"{movie_title} (Your rating: {top_rated.iloc[i]['rating']})")
        
        st.subheader("Your Recommendation:")
        cols = st.columns(5)
        with st.spinner("Generating recommendations..."):
            recommended_ids = model.hybrid_recommendation(user_id,movies_df,None,5)
            print(recommended_ids)
            for i, (movie_id,score) in enumerate(recommended_ids):
                title_row = movies_df[movies_df['movieId'] == movie_id]
                movie_title = title_row['title'].iloc[0] if not title_row.empty else "Unknown"
                

                with cols[i]:
                    cleaned_title = clean_title(movie_title)
                    poster_url = get_movie_poster(cleaned_title, TMDB_API_KEY)
                    st.image(poster_url, use_container_width=True)
                    st.caption(f"{movie_title}\n⭐ Ratings: {score:.2f}")
        
        st.markdown("---")
        st.subheader("🎯 Filter Recommendations by Genre")

        # Extract all unique genres from the dataset
        all_genres = sorted(
            set(g for genre_list in movies_df['genres'].dropna().str.split('|') for g in genre_list)
        )

        selected_genre = st.selectbox("Choose a genre:", ["-- Select Genre --"] + all_genres)

        if selected_genre != "-- Select Genre --":
            genre_movies = get_movies_by_genre(movies_df, selected_genre)

            if genre_movies.empty:
                st.warning("No movies found for this genre.")
            else:
                st.subheader(f"📽️ Recommended '{selected_genre}' Movies:")
                genre_movie_ids = genre_movies['movieId'].tolist()
                recommendations = model.hybrid_recommendation(user_id, movies_df, None, 20)

                # Filter recommended movies by genre
                genre_recommendations = [(mid, score) for (mid, score) in recommendations if mid in genre_movie_ids][:5]

                if genre_recommendations:
                    cols = st.columns(5)
                    for i, (movie_id, score) in enumerate(genre_recommendations):
                        title_row = movies_df[movies_df['movieId'] == movie_id]
                        movie_title = title_row['title'].iloc[0] if not title_row.empty else "Unknown"
                        cleaned_title = clean_title(movie_title)
                        poster_url = get_movie_poster(cleaned_title, TMDB_API_KEY)
                        with cols[i]:
                            st.image(poster_url, use_container_width=True)
                            st.caption(f"{movie_title}\n⭐ Score: {score:.2f}")
        else:
            st.info("No recommendations in this genre.")
        
        if st.button("Go Back to Main Menu"):
            st.session_state.user_id = None
            st.session_state.is_new_user = False
            st.rerun()