from django.urls import path
from . import views

app_name = "accounts"
urlpatterns = [
    path('', views.signup, name="signup"), # GET, POST
    path('login/', views.login, name="login"),
    path('', views.signout, name="signout"), # DELETE
    path('time/', views.time, name="time"), # POST
]