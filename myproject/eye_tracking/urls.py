from django.urls import path
from .views import webcam

urlpatterns = [
    path('webcam/', webcam, name='webcam'),
]
