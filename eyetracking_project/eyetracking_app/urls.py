from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('live_feed/', views.live_feed, name='live_feed'), 
    path('dynamic_stream/', views.dynamic_stream, name='dynamic_stream'),
    path('toggle_recording/', views.toggle_recording, name='toggle_recording'),


]
