from django.urls import path
from . import views

app_name = "detections"
urlpatterns = [
    path('neck/', views.check_neck, name="check_neck"), # POST : 거북목 설정 (초기 4회) & 거북목 체크 (5회 이상)
    path('blink/', views.check_blink, name="check_blink"), # POST : 눈 깜박임 체크
]