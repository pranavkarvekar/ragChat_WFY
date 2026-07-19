from django.urls import path
# pyrefly: ignore [missing-import]
from . import views

urlpatterns = [
    path('api/web/', views.api_web_chat, name='api_web_chat'),
    path('api/youtube/', views.api_youtube_chat, name='api_youtube_chat'),
    path('api/files/', views.api_file_chat, name='api_file_chat'),
    path('api/status/', views.api_status, name='api_status'),
    path('api/chat/', views.api_chat, name='api_chat'),
]


