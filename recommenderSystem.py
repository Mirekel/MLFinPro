# Hybrid Movie Recommender System
# Combining User-based CF, Item-based CF, and Content-based Filtering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
from sklearn.preprocessing import MinMaxScaler
import warnings
import streamlit as st
import pickle
import os
warnings.filterwarnings('ignore')


# Set random seed for reproducibility
np.random.seed(42)

# Load the datasets
movies = pd.read_csv('./datasets/movies.csv')
ratings = pd.read_csv('./datasets/ratings.csv', nrows=100000)

# def normalize_dict(d):
#     if not d:
#         return {}
#     values = np.array(list(d.values()))
#     min_val, max_val = values.min(), values.max()
#     if min_val == max_val:
#         return {k: 0.5 for k in d}  # arbitrary mid value
#     return {k: (v - min_val) / (max_val - min_val) for k, v in d.items()}

def load_and_preprocess_data(new_user_id = None):
    print("Loading datasets...")
    print(f"Movies dataset shape: {movies.shape}")
    print(f"Ratings dataset shape: {ratings.shape}")
    print("\nMovies dataset preview:")
    print(movies.head())
    print("\nRatings dataset preview:")
    print(ratings.head())

    print(f"Unique users: {ratings['userId'].nunique()}")
    print(f"Unique movies: {ratings['movieId'].nunique()}")
    print(f"Total ratings: {len(ratings)}")
    print(f"Rating range: {ratings['rating'].min()} - {ratings['rating'].max()}")
    print(f"Average rating: {ratings['rating'].mean():.2f}")

    data = ratings.merge(movies, on='movieId', how='left')
    print(f"\nMerged dataset shape: {data.shape}")

    
    min_user_ratings = 20
    min_movie_ratings = 10

    user_counts = ratings['userId'].value_counts()
    movie_counts = ratings['movieId'].value_counts()

    active_users = user_counts[user_counts >= min_user_ratings].index
    popular_movies = movie_counts[movie_counts >= min_movie_ratings].index
    
    if new_user_id is not None:
    # Always include the new user, even if they don't meet the threshold
        active_users = active_users.union(pd.Index([new_user_id]))

    filtered_ratings = ratings[
        (ratings['userId'].isin(active_users)) & 
        (ratings['movieId'].isin(popular_movies))
    ]

    print(f"\nAfter filtering:")
    print(f"Active users: {filtered_ratings['userId'].nunique()}")
    print(f"Popular movies: {filtered_ratings['movieId'].nunique()}")
    print(f"Remaining ratings: {len(filtered_ratings)}")

    # Create train-test split
    train_data, test_data = train_test_split(filtered_ratings, test_size=0.2, random_state=42, stratify=filtered_ratings['userId'])

    print(f"\nTrain set: {len(train_data)} ratings")
    print(f"Test set: {len(test_data)} ratings")
    
    return train_data, test_data

def generate_user_item_matrices(train_data,test_data):
    def create_user_item_matrix(data):
        return data.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

    train_matrix = create_user_item_matrix(train_data)
    test_matrix = create_user_item_matrix(test_data)

    print(f"\nTrain matrix shape: {train_matrix.shape}")
    print(f"Matrix sparsity: {(train_matrix == 0).sum().sum() / (train_matrix.shape[0] * train_matrix.shape[1]) * 100:.2f}%")    

    return train_matrix, test_matrix
   

#modes
class UserBasedCF:
    def __init__(self, k=50):
        self.k = k

    def fit(self, train_matrix):
        self.train_matrix = train_matrix
        self.user_ids = train_matrix.index.tolist()
        self.user_idx_map = {u: i for i, u in enumerate(self.user_ids)}
        self.user_similarity = cosine_similarity(train_matrix.values)
        np.fill_diagonal(self.user_similarity, 0)

    def predict(self, user_id, movie_id):
        if user_id not in self.train_matrix.index or movie_id not in self.train_matrix.columns:
            return np.nanmean(self.train_matrix.values)

        u_idx = self.user_idx_map[user_id]
        sims = self.user_similarity[u_idx]
        movie_col = self.train_matrix[movie_id]
        rated = movie_col[movie_col > 0]

        if rated.empty:
            return np.nanmean(self.train_matrix.values)

        indices = [self.user_idx_map[uid] for uid in rated.index]
        ratings = rated.values
        sim_scores = sims[indices]

        top_k = np.argsort(sim_scores)[-self.k:]
        top_ratings = ratings[top_k]
        top_sims = sim_scores[top_k]

        denom = np.sum(np.abs(top_sims))
        return np.dot(top_ratings, top_sims) / denom if denom else np.nanmean(self.train_matrix.values)

class ItemBasedCF:
    def __init__(self, k=50):
        self.k = k

    def fit(self, train_matrix):
        self.train_matrix = train_matrix
        self.movie_ids = train_matrix.columns.tolist()
        self.movie_idx_map = {m: i for i, m in enumerate(self.movie_ids)}
        self.item_similarity = cosine_similarity(train_matrix.T.values)
        np.fill_diagonal(self.item_similarity, 0)

    def predict(self, user_id, movie_id):
        if user_id not in self.train_matrix.index or movie_id not in self.train_matrix.columns:
            return np.nanmean(self.train_matrix.values)

        m_idx = self.movie_idx_map[movie_id]
        sims = self.item_similarity[m_idx]
        user_row = self.train_matrix.loc[user_id]
        rated = user_row[user_row > 0]

        if rated.empty:
            return np.nanmean(self.train_matrix.values)

        indices = [self.movie_idx_map[mid] for mid in rated.index]
        ratings = rated.values
        sim_scores = sims[indices]

        top_k = np.argsort(sim_scores)[-self.k:]
        top_ratings = ratings[top_k]
        top_sims = sim_scores[top_k]

        denom = np.sum(np.abs(top_sims))
        return np.dot(top_ratings, top_sims) / denom if denom else np.nanmean(self.train_matrix.values)

class ContentBasedCF:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english')

    def fit(self, train_matrix, movies_df):
        self.train_matrix = train_matrix
        self.movie_ids = train_matrix.columns.tolist()
        movies = movies_df[movies_df['movieId'].isin(self.movie_ids)].copy()
        movies['genres'] = movies['genres'].fillna('Unknown')
        movies['combined'] = movies['title'] + " " + movies['genres']
        self.movie_features = self.tfidf.fit_transform(movies['combined'])
        self.movie_similarity = cosine_similarity(self.movie_features)
        self.movie_idx_map = {mid: i for i, mid in enumerate(movies['movieId'])}

    def predict(self, user_id, movie_id):
        if user_id not in self.train_matrix.index or movie_id not in self.movie_idx_map:
            return np.nanmean(self.train_matrix.values)

        user_ratings = self.train_matrix.loc[user_id]
        rated = user_ratings[user_ratings > 0]

        if rated.empty:
            return np.nanmean(self.train_matrix.values)

        m_idx = self.movie_idx_map[movie_id]
        total, weight = 0, 0

        for rid, rating in rated.items():
            if rid in self.movie_idx_map:
                r_idx = self.movie_idx_map[rid]
                sim = self.movie_similarity[m_idx, r_idx]
                total += sim * rating
                weight += sim

        return total / weight if weight else np.nanmean(self.train_matrix.values)
    
class HybridRecommender:
    def __init__(self, user_weight=0.3, item_weight=0.4, content_weight=0.3):
        self.user_weight = user_weight
        self.item_weight = item_weight
        self.content_weight = content_weight
        self.user_cf = UserBasedCF()
        self.item_cf = ItemBasedCF()
        self.content_cf = ContentBasedCF()

    def fit(self, train_matrix, movies_df):
        self.user_cf.fit(train_matrix)
        self.item_cf.fit(train_matrix)
        self.content_cf.fit(train_matrix, movies_df)

    def predict(self, user_id, movie_id):
        preds = np.array([
            self.user_cf.predict(user_id, movie_id),
            self.item_cf.predict(user_id, movie_id),
            self.content_cf.predict(user_id, movie_id)
        ])
        weights = np.array([self.user_weight, self.item_weight, self.content_weight])
        return np.dot(preds, weights)

    def recommend_movies(self, user_id, n=10):
        if user_id not in self.user_cf.train_matrix.index:
            return []

        user_ratings = self.user_cf.train_matrix.loc[user_id]
        unrated = user_ratings[user_ratings == 0].index

        predictions = [(m, self.predict(user_id, m)) for m in unrated]
        return sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    
    
    def recommend_by_title(self, movies_df, title, k=5):
        matched = movies_df[movies_df['title'].str.lower().str.strip() == title.lower().strip()]
        if matched.empty:
            raise ValueError("Movie title not found.")

        movie_id = matched['movieId'].values[0]

        if movie_id not in self.content_cf.movie_idx_map:
            raise ValueError("Movie ID not found in content-based model.")

        idx = self.content_cf.movie_idx_map[movie_id]

        sim_scores = list(enumerate(self.content_cf.movie_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:k+1]

        movie_ids = list(self.content_cf.movie_idx_map.keys())

        recommended_movie_ids = [movie_ids[i] for i, _ in sim_scores]

        return recommended_movie_ids

    def hybrid_recommendation(self, user_id, movies_df, title=None, n=5, user_weight = 0.6 , item_weight = 0.2 , content_weight = 0.2):
        #1--UserBasedCF
        if user_id in self.user_cf.train_matrix.index:
            user_ratings = self.user_cf.train_matrix.loc[user_id]
            unrated = user_ratings[user_ratings == 0].index
            user_cf_scores = {m : self.user_cf.predict(user_id,m) for m in unrated} 
        else: 
            user_cf_scores = {}   
        #2--ItemBasedCF
        if user_id in self.item_cf.train_matrix.index:
            item_cf_scores = {m: self.item_cf.predict(user_id,m) for m in unrated}
        else:
            item_cf_scores = {}
        #3--ContentBasedCF
        if title is not None:
            try:
                content_ids = self.recommend_by_title(movies_df,title, k=10)
            except ValueError as e:
                print(e)
                content_ids = []
        else:
            content_ids = []
            
        content_scores = {}
        for rank, movie_id in enumerate(content_ids):
            content_scores[movie_id] = (n - rank) / n #top n movies
            
        if content_scores:
            values = np.array(list(content_scores.values())).reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(1, 5))
            scaled = scaler.fit_transform(values).flatten()
            
            content_scores = {
                k: scaled[i] for i, k in enumerate(content_scores.keys())
            }
        else:
            content_scores = {}
        
        #4.Merge all ids
        all_movies_ids = set(user_cf_scores.keys() | item_cf_scores.keys() | content_scores.keys())
        
        #5.Combine Scores
        hybrid_scores = {}
        for movie_id in all_movies_ids:
            u_score = user_cf_scores.get(movie_id,0)
            i_score = item_cf_scores.get(movie_id,0)
            c_score = content_scores.get(movie_id,0)
            
            hybrid_scores[movie_id] =  user_weight * u_score + item_weight * i_score + content_weight * c_score
            
        recommended = sorted(hybrid_scores.items(), key=lambda x:x[1], reverse=True)[:n]
        # st.write("recommended data ids",recommended)
        # st.write(f"Is user {user_id} in user_cf train matrix? ", user_id in self.user_cf.train_matrix.index)
        # st.write(f"Is user {user_id} in item_cf train matrix? ", user_id in self.item_cf.train_matrix.index)
        
        return  recommended
                       
class RecommenderEvaluator:
        def __init__(self, threshold=3.5):
            self.threshold = threshold  # Rating threshold for relevance
            
        def precision_at_k(self, actual, predicted, k=10):
            """Calculate Precision@K"""
            if k > len(predicted):
                k = len(predicted)
            
            predicted_k = predicted[:k]
            relevant_predicted = sum(1 for item in predicted_k if item in actual)
            
            return relevant_predicted / k if k > 0 else 0
        
        def recall_at_k(self, actual, predicted, k=10):
            """Calculate Recall@K"""
            if k > len(predicted):
                k = len(predicted)
            
            predicted_k = predicted[:k]
            relevant_predicted = sum(1 for item in predicted_k if item in actual)
            
            return relevant_predicted / len(actual) if len(actual) > 0 else 0
        
        def ndcg_at_k(self, actual, predicted, k=10):
            """Calculate NDCG@K"""
            if k > len(predicted):
                k = len(predicted)
            
            predicted_k = predicted[:k]

            dcg = 0
            for i, item in enumerate(predicted_k):
                if item in actual:
                    dcg += 1 / np.log2(i + 2)

            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(actual), k)))
            
            return dcg / idcg if idcg > 0 else 0
        
        def mean_average_precision(self, actual_dict, predicted_dict, k=10):
            """Calculate Mean Average Precision"""
            ap_scores = []
            
            for user_id in actual_dict:
                if user_id in predicted_dict:
                    actual = actual_dict[user_id]
                    predicted = predicted_dict[user_id][:k]
                    
                    if len(actual) == 0:
                        continue
                    
                    # Calculate Average Precision
                    precision_scores = []
                    relevant_count = 0
                    
                    for i, item in enumerate(predicted):
                        if item in actual:
                            relevant_count += 1
                            precision_scores.append(relevant_count / (i + 1))
                    
                    ap = sum(precision_scores) / len(actual) if len(actual) > 0 else 0
                    ap_scores.append(ap)
            
            return np.mean(ap_scores) if ap_scores else 0
        
        def evaluate_model(self, model, test_data, train_matrix, k=10):
            test_users = test_data['userId'].unique()
            
            actual_relevant = {}
            predicted_items = {}
            
            print(f"Evaluating model for {len(test_users)} users...")
            
            for i, user_id in enumerate(test_users[:100]):  # Limit to 100 users for faster evaluation
                if i % 20 == 0:
                    print(f"Progress: {i}/{min(100, len(test_users))}")

                user_test_data = test_data[test_data['userId'] == user_id]
                actual_relevant[user_id] = set(user_test_data[user_test_data['rating'] >= self.threshold]['movieId'])

                recommendations = model.recommend_movies(user_id, n=k)
                predicted_items[user_id] = [movie_id for movie_id, _ in recommendations]
            
            precision_scores = []
            recall_scores = []
            ndcg_scores = []
            
            for user_id in actual_relevant:
                if user_id in predicted_items:
                    actual = list(actual_relevant[user_id])
                    predicted = predicted_items[user_id]
                    
                    precision_scores.append(self.precision_at_k(actual, predicted, k))
                    recall_scores.append(self.recall_at_k(actual, predicted, k))
                    ndcg_scores.append(self.ndcg_at_k(actual, predicted, k))
            
            map_score = self.mean_average_precision(actual_relevant, predicted_items, k)
            
            results = {
                'Precision@K': np.mean(precision_scores),
                'Recall@K': np.mean(recall_scores),
                'NDCG@K': np.mean(ndcg_scores),
                'MAP': map_score
            }
            
            return results

def model_fitting_evaluation(train_matrix):
    global movies
    
    hybrid_model = HybridRecommender(user_weight=0.3, item_weight=0.4, content_weight=0.3)
    hybrid_model.fit(train_matrix, movies)

   
    return hybrid_model
    
def model_testing():
    hybrid_model = model_fitting_evaluation()
    
    sample_user = 50
    recommendations = hybrid_model.recommend_movies(sample_user, n=10)

    print(f"\nTop 10 movie recommendations for User {sample_user}:")
    print("-" * 60)
    for i, (movie_id, pred_rating) in enumerate(recommendations, 1):
        movie_title = movies[movies['movieId'] == movie_id]['title'].iloc[0] if len(movies[movies['movieId'] == movie_id]) > 0 else "Unknown"
        print(f"{i:2d}. {movie_title} (Predicted Rating: {pred_rating:.2f})")
    
def add_user(user_id):
    new_row = {"userId": user_id, "movieId": None, "rating": None}
    global ratings  
    ratings = pd.concat([ratings, pd.DataFrame([new_row])], ignore_index=True)

def add_ratings(user_id, ratings_dict):
    global ratings
    new_rows = []
    for movie_id_str, rating in ratings_dict.items():
        movie_id = int(movie_id_str)
        new_rows.append({"userId": user_id, "movieId": movie_id, "rating": rating})

    ratings.dropna(inplace=True)
    ratings = pd.concat([ratings, pd.DataFrame(new_rows)], ignore_index=True)
    
    ratings.to_csv("datasets/ratings.csv", index=False)

def reload_and_retrain_model(new_user_id):
    global ratings, hybrid_model
    
    ratings = pd.read_csv('./datasets/ratings.csv')
    st.spinner('Retraining model...')
    train_data,test_data = load_and_preprocess_data(new_user_id)
    train_matrix, test_matrix = generate_user_item_matrices(train_data, test_data)
    hybrid_model = model_fitting_evaluation(train_matrix)
    st.success('Model retraining done...')

    with open("model/recommenderNew.pkl", "wb") as f:
        pickle.dump(hybrid_model, f)
        st.success('Model saved to disk.')
    
def update_model():
    with open("model/recommenderNew.pkl", "rb") as f:
        global hybrid_model
        pickle.load(f)
        
    with open("model/recommender.pkl", "wb") as f:
        st.spinner('updating original model..')
        pickle.dump(hybrid_model,f )
        st.success("Model successfully updated and saved.")
 
def get_movies_by_genre(movies_df, genre):
    return movies_df[movies_df['genres'].str.contains(genre, case=False, na=False)]