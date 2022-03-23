from django.urls import path
from . import views

app_name = "accounts"
urlpatterns = [
    path('', views.accounts, name="accounts"), # POST: 회원가입(DB), DELETE : 회원탈퇴
    path('time/', views.time, name="time"), # GET: 조회, PUT: 수정
]