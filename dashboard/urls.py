from django.urls import path

from dashboard import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/departments/", views.api_departments, name="api_departments"),
    path("api/companies/", views.api_companies, name="api_companies"),
    path("api/plants/", views.api_plants, name="api_plants"),
    path("api/predict/", views.api_predict, name="api_predict"),
]
