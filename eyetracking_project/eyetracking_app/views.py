from django.shortcuts import render

def index(request):
    return render(request, 'eyetracking_app/index.html')
