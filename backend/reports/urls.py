from django.urls import path
from . import views

app_name = "reports"
urlpatterns = [
    path('', views.reports, name="/") # [POST] 저장, [GET] 전체 내보내기
]