from django.urls import path
from . import views
from django.contrib import admin


urlpatterns = [
    
    path('admin/', admin.site.urls),

    path('', views.movie_search, name='index'),
]
