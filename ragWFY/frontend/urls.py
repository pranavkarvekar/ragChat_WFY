from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('web/', views.web_chat, name='web_chat'),
    path('youtube/', views.youtube_chat, name='youtube_chat'),
    path('files/', views.file_chat, name='file_chat'),
]


