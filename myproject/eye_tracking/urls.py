from django.urls import path
from . import views

urlpatterns = [
    path('webcam/', views.webcam, name='webcam'),
]
