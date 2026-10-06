
from django import forms

class MovieSearchForm(forms.Form):
    movie_name = forms.CharField(
        max_length=200,
        required=True,
        label='',
        widget=forms.TextInput(attrs={
            'placeholder': 'Search any movie (e.g. Inception, Avatar, The Dark Knight)...',
            'class': 'search-input-field',
            'id': 'movie-search-input',
            'autocomplete': 'off',
            'list': 'movie-datalist'
        })
    )