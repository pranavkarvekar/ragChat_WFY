from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views.decorators.http import require_http_methods
from django.contrib.auth.models import User
from .models import UserProfile


@login_required(login_url='login')
@require_http_methods(["GET"]) 
def home(request):
    return render(request, 'home.html')


@require_http_methods(["GET", "POST"]) 
def login_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        # Try to find user by email first, then by username
        try:
            user = User.objects.get(email=email)
            user = authenticate(request, username=user.username, password=password)
        except User.DoesNotExist:
            user = authenticate(request, username=email, password=password)
        
        if user is not None:
            login(request, user)
            messages.success(request, f'Welcome back, {user.first_name or user.username}!')
            return redirect('home')
        messages.error(request, 'Invalid email or password')
        return render(request, 'login.html')
    return render(request, 'login.html')


@require_http_methods(["GET", "POST"]) 
def register_view(request):
    if request.method == 'POST':
        first_name = request.POST.get('first_name')
        email = request.POST.get('email')
        username = request.POST.get('username')
        password = request.POST.get('password')
        phone_number = request.POST.get('phone_number', '')
        
        # Validation
        if not all([first_name, email, username, password]):
            messages.error(request, 'All fields are required')
            return render(request, 'login.html')
        
        if User.objects.filter(email=email).exists():
            messages.error(request, 'Email already exists')
            return render(request, 'login.html')
        
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists')
            return render(request, 'login.html')
        
        try:
            # Create user
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password,
                first_name=first_name
            )
            
            # Create user profile
            UserProfile.objects.create(user=user, phone_number=phone_number)
            
            # Auto-login after registration
            login(request, user)
            messages.success(request, f'Account created successfully! Welcome, {user.first_name}!')
            return redirect('home')
            
        except Exception as e:
            messages.error(request, f'Error creating account: {str(e)}')
            return render(request, 'login.html')
    
    return render(request, 'login.html')


@require_http_methods(["GET", "POST"]) 
def logout_view(request):
    logout(request)
    messages.info(request, 'You have been logged out successfully')
    return redirect('login')


@login_required(login_url='login')
def web_chat(request):
    return render(request, 'web_chat.html')


@login_required(login_url='login')
def youtube_chat(request):
    return render(request, 'youtube_chat.html')


@login_required(login_url='login')
def file_chat(request):
    return render(request, 'file_chat.html')

# Create your views here.
