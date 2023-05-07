# Import necessary libraries
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import JsonResponse

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# Define the view function that will be called when the API endpoint is accessed
def MainView(request, movie):
    # Load the movies and ratings datasets
    movies = pd.read_csv('ml-latest-small/movies.csv')
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    
    # Merge the two datasets on the 'movieId' column
    data = pd.merge(ratings, movies, on='movieId')
    
    # Create a pivot table with 'movieId' as the index, 'userId' as the columns, and 'rating' as the values.
    # Fill missing values with 0
    pivot_table = data.pivot_table(index='movieId', columns='userId', values='rating').fillna(0)
    
    # Convert the pivot table to a sparse matrix to save memory
    sparse_matrix = csr_matrix(pivot_table.values)
    
    # Define the nearest neighbors model with 'cosine' similarity metric and 'brute' algorithm
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    
    # Fit the model to the sparse matrix
    model.fit(sparse_matrix)
    
    # Get the movieId of the input movie title
    toy_story_id = movies[movies['title'] == movie]['movieId'].iloc[0]
    
    # Get the index of the input movie in the pivot table
    toy_story_index = pivot_table.index.get_loc(toy_story_id)
    
    # Find the 10 nearest neighbors to the input movie based on user ratings
    # Get the distances and indices of the neighbors
    distances, indices = model.kneighbors(pivot_table.iloc[toy_story_index, :].values.reshape(1, -1), n_neighbors=11)
    
    # Get the titles of the 10 similar movies
    similar_movies = []
    for index in indices.flatten()[1:]:
        movie_title = movies[movies['movieId'] == pivot_table.iloc[index].name]['title'].iloc[0]
        similar_movies.append(movie_title)

    # Create a dictionary with the similar movies list and return as JSON response
    response_data = {'movies': similar_movies}
    return JsonResponse(response_data)
