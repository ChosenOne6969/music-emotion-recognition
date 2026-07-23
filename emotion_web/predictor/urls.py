from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict_emotion, name='predict_emotion'),
    path('about/', views.about, name='about'),
    path('upload/', views.upload_predict, name='upload_predict'),
]