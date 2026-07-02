from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('download/csv/', views.download_csv, name='download_csv'),
    path('download/excel/', views.download_excel, name='download_excel'),
    path('download/pdf/', views.download_pdf, name='download_pdf'),
]
