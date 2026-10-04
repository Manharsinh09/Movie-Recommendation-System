from django.shortcuts import render
import requests
from .forms import MovieSearchForm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.shortcuts import render
from django.conf import settings
import os
from django.conf import settings
import pandas as pd

OMDB_API_KEY = settings.OMDB_API_KEY  
OMDB_API_URL = 'http://www.omdbapi.com/'
# core/views.py

file_path = os.path.join(settings.BASE_DIR,'recommendation', 'static', 'data', 'movies.json')
movie_data = pd.read_json(file_path).to_dict(orient='records')


def custom_error_view(request, exception=None):
    status_code = getattr(exception, 'status_code', 500)
    context = {
        'status_code': status_code,
    }
    return render(request, 'recommendation/error-page.html', context, status=status_code)

def fetch_movie_details(movie_name):
    response = requests.get(OMDB_API_URL, params={'t': movie_name, 'apikey': OMDB_API_KEY})
    data = response.json()

    if data['Response'] == 'True':
        return {
            'title': data['Title'],
            'description': data['Plot'],
            'year': data['Year'],
            'poster': data['Poster'],
        }
    else:
        return None  

# movie_data = [
# {"movie_id":25, "title": "Jersey", "tages": "A former cricketer decides to make a comeback to fulfill his son's wish for a jersey, challenging his past regrets and his passion for cricket. Starring Shahid Kapoor, Mrunal Thakur, and Pankaj Kapur."},
# ]

def recommend_movies(movie_name, movie_data):
    movie_titles = [movie['title'] for movie in movie_data]
    movie_descriptions = [movie['tages'] for movie in movie_data]
    
    movie_name_lower = movie_name.lower()

    movie_index = -1
    for idx, title in enumerate(movie_titles):
        if movie_name_lower in title.lower():  
            movie_index = idx
            break

    if movie_index == -1:
        return []  
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(movie_descriptions)
    
    cosine_similarities = cosine_similarity(tfidf_matrix[movie_index], tfidf_matrix).flatten()
    similar_indices = cosine_similarities.argsort()[::-1][:6] 
    recommended_movies = []
    
    for i in similar_indices:
        movie = movie_titles[i]
        movie_details = fetch_movie_details(movie)
        poster_url = movie_details.get('poster', 'https://via.placeholder.com/150')
        
        recommended_movies.append((movie, cosine_similarities[i], poster_url))
    
    return recommended_movies

def movie_details(request, movie_name):
    movie_details = fetch_movie_details(movie_name) 
    recommended_movies = recommend_movies(movie_name, movie_data)
    
    return render(request, 'index.html', {
        'movie_details': movie_details,
        'recommended_movies': recommended_movies
    })



def movie_search(request):
    form = MovieSearchForm()
    recommendations = []
    movie_details = None

    if request.method == 'POST':
        form = MovieSearchForm(request.POST)
        if form.is_valid():
            movie_name = form.cleaned_data['movie_name']
            
            # Fetch movie details
            movie_details = fetch_movie_details(movie_name)
            
            # Get 5 recommended movies
            if movie_details:
                recommendations = recommend_movies(movie_name, movie_data)

    return render(request, 'recommendation/index.html', {
        'form': form,
        'recommendations': recommendations,
        'movie_details': movie_details
    })