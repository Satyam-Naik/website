from django.urls import path
from . import views

app_name = "website"
urlpatterns = [
    path('', views.get_details, name="details"),
    path('home', views.home, name="home"),
    path('clear_session', views.clear_session, name="clear_session"),
    path('optimisation', views.optimisation, name="optimisation"),
    path('knn', views.knn, name="knn")
]