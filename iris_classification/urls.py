from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('knn/', include('knn.urls')),
]
