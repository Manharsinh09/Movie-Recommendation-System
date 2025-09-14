from django.shortcuts import render
import requests
from .forms import MovieSearchForm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.conf import settings
OMDB_API_KEY = settings.OMDB_API_KEY  
OMDB_API_URL = 'http://www.omdbapi.com/'
# core/views.py

from django.shortcuts import render

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

movie_data = [
    {"title": "The Shawshank Redemption", "description": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency. Starring Tim Robbins and Morgan Freeman.", "poster": "poster_url_1"},
    {"title": "The Godfather", "description": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son. Starring Marlon Brando, Al Pacino, and James Caan.", "poster": "poster_url_2"},
    {"title": "The Dark Knight", "description": "When the menace known as The Joker emerges from his mysterious past, he wreaks havoc and chaos on the people of Gotham. Starring Christian Bale, Heath Ledger, and Aaron Eckhart.", "poster": "poster_url_3"},
    {"title": "Pulp Fiction", "description": "The lives of two mob hitmen, a boxer, a gangster's wife, and a pair of diner bandits intertwine in four tales of violence and redemption. Starring John Travolta, Uma Thurman, and Samuel L. Jackson.", "poster": "poster_url_4"},
    {"title": "Forrest Gump", "description": "The presidencies of Kennedy and Johnson, the events of Vietnam, the Civil Rights Movement, and other historical events unfold from the perspective of an Alabama man with an extraordinary history. Starring Tom Hanks and Robin Wright.", "poster": "poster_url_5"},
    {"title": "Sholay", "description": "Two criminals are hired by a retired police officer to capture the ruthless bandit leader, Gabbar Singh. Starring Amitabh Bachchan, Dharmendra, and Hema Malini.", "poster": "poster_url_6"},
    {"title": "Dilwale Dulhania Le Jayenge", "description": "A young man and woman fall in love during a trip to Europe but are torn between family obligations and their relationship. Starring Shah Rukh Khan and Kajol.", "poster": "poster_url_7"},
    {"title": "Lagaan", "description": "In 1893, a group of Indian villagers rise up against British colonial rule when they are challenged to a game of cricket. Starring Aamir Khan, Gracy Singh, and Paul Blackthorne.", "poster": "poster_url_8"},
    {"title": "3 Idiots", "description": "Three engineering students set out to discover the real meaning of life and challenge the rigid education system in India. Starring Aamir Khan, R. Madhavan, and Sharman Joshi.", "poster": "poster_url_9"},
    {"title": "Dangal", "description": "The story of a father who trains his daughters to become world-class wrestlers, defying societal expectations. Starring Aamir Khan, Fatima Sana Shaikh, and Sanya Malhotra.", "poster": "poster_url_11"},
    {"title": "Queen", "description": "A young woman embarks on a solo honeymoon trip to Europe after her wedding is called off, finding herself in the process. Starring Kangana Ranaut and Rajkummar Rao.", "poster": "poster_url_12"},
    {"title": "Zindagi Na Milegi Dobara", "description": "Three friends take a road trip across Spain, where they face their fears and come to terms with their past. Starring Hrithik Roshan, Farhan Akhtar, and Abhay Deol.", "poster": "poster_url_13"},
    {"title": "Mughal-e-Azam", "description": "A historical epic that depicts the love story between Prince Salim and Anarkali amidst the royal palace of the Mughal Empire. Starring Prithviraj Kapoor, Dilip Kumar, and Madhubala.", "poster": "poster_url_14"},
    {"title": "Pathaan", "description": "A spy thriller where an Indian spy is forced to battle against an international threat to prevent a global disaster. Starring Shah Rukh Khan, Deepika Padukone, and John Abraham.", "poster": "poster_url_15"},
    {"title": "RRR", "description": "A fictional story about two revolutionaries in colonial India who fight against the British Empire for freedom. Starring N. T. Rama Rao Jr., Ram Charan, and Alia Bhatt.", "poster": "poster_url_16"},
    {"title": "Brahmāstra: Part One – Shiva", "description": "A young man discovers his hidden powers and joins forces with ancient guardians to fight evil forces. Starring Ranbir Kapoor, Alia Bhatt, and Amitabh Bachchan.", "poster": "poster_url_17"},
    {"title": "Gangubai Kathiawadi", "description": "The true story of Gangubai, a woman who rises to become one of the most feared and powerful women in the Mumbai underworld. Starring Alia Bhatt and Ajay Devgn.", "poster": "poster_url_18"},
    {"title": "Laal Singh Chaddha", "description": "An adaptation of the classic film Forrest Gump, following the life of a simple man who unknowingly influences historical events. Starring Aamir Khan and Kareena Kapoor.", "poster": "poster_url_19"},
    {"title": "Drishyam 2", "description": "The sequel to the critically acclaimed thriller where a man tries to protect his family from the consequences of their dark secrets. Starring Ajay Devgn, Tabu, and Shriya Saran.", "poster": "poster_url_20"},
    {"title": "Kantara", "description": "A rural action drama set in the backdrop of forest-dwelling tribes and their battles against external forces. Starring Rishab Shetty, Sapthami Gowda, and Kishore." , "poster": "poster_url_21"},
    {"title": "Shershaah", "description": "A biographical war film based on the life of Captain Vikram Batra, a hero of the Kargil War. Starring Sidharth Malhotra, Kiara Advani, and Sharib Hashmi.", "poster": "poster_url_22"},
    {"title": "Rocketry: The Nambi Effect", "description": "The inspiring true story of Nambi Narayanan, a scientist who was falsely accused of espionage and later proved to be a hero. Starring R. Madhavan and Simran.", "poster": "poster_url_23"},
    {"title": "Jersey", "description": "A former cricketer decides to make a comeback to fulfill his son's wish for a jersey, challenging his past regrets and his passion for cricket. Starring Shahid Kapoor, Mrunal Thakur, and Pankaj Kapur.", "poster": "poster_url_24"},
]


def recommend_movies(movie_name, movie_data):
    movie_titles = [movie['title'] for movie in movie_data]
    movie_descriptions = [movie['description'] for movie in movie_data]
    
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