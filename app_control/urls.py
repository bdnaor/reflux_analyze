from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^get_models$', views.get_models),
    url(r'^add_model$', views.add_model),
    url(r'^predict_images$', views.predict_images),
]
