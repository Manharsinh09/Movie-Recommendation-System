
from django import forms

class MovieSearchForm(forms.Form):
    movie_name = forms.CharField(max_length=200, label='Search for a movie')