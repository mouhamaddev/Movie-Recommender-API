from django.urls import path
from .views import MainView

urlpatterns = [
    path('api/<str:movie>/', MainView, name='main'),
]