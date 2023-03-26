from django.shortcuts import render

def webcam(request):
    return render(request, 'eye_tracking/webcam.html')
