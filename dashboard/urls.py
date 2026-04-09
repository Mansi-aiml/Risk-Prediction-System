from django.urls import path

from dashboard import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/departments/", views.api_departments, name="api_departments"),
    path("api/predict/", views.api_predict, name="api_predict"),
]
