from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # URL pattern for index view
    path('search/', views.search, name='search'),
    path('predict/<str:ticker_value>/<str:number_of_days>/', views.predict, name='predict'),
    path('ticker/', views.ticker, name='ticker'),
]