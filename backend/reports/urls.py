from django.urls import path
from . import views

app_name = "reports"
urlpatterns = [
    path('', views.reports, name="report_list") # POST : 단일 저장, GET : 전체 조회
]