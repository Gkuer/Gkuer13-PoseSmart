from django.urls import path
from . import views

app_name = "detections"
urlpatterns = [
    path('', views.check_neck, name="check_neck"), # POST : 거북목 설정 & 거북목 체크
]