from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static

# for registration view
from django.conf.urls import url

# for password reset, 
from .forms import UserPasswordResetForm, UserSetPasswordForm
from django.contrib.auth.views import PasswordResetView, PasswordResetConfirmView

urlpatterns = [
    #path('', views.home),
    #path('', views.home ,name='home'),
    path('', views.LandingView.as_view(), name='landing'),    
    path('<str:stay>', views.LandingView.as_view(), name='landing'),
    path('dashboard/', views.DashboardView.as_view(), name='dashboard'),    
    path('datasets/', views.DatasetsView.as_view(), name="datasets"),
    path('datasets/<int:dataset_id>/detail/', views.DatasetDetailView.as_view(), name="dataset_detail"),
    path('models/', views.ModelListView.as_view(), name="model_listview"),
    path('results/<int:mlmodelid>', views.ResultsView.as_view(), name="results"),
    path('accounts/registrieren', views.Registration, name="registration"),
    path('einstellungen/', views.ConfigView.as_view(), name="config"),
    # custom form for password_reset
    path('accounts/password_reset/', PasswordResetView.as_view(
        template_name='registration/password_reset_form.html',
        form_class=UserPasswordResetForm),name='password_reset'),
    # custom form for password change    
    path('accounts/reset/<uidb64>/<token>/', PasswordResetConfirmView.as_view(
        template_name='registration/password_reset_confirm.html',
        form_class=UserSetPasswordForm),name='password_reset_confirm'),    
    
    url(r'^activate/(?P<uidb64>[0-9A-Za-z_\-]+)/(?P<token>[0-9A-Za-z]{1,13}-[0-9A-Za-z]{1,20})/$',
        views.Activate.as_view(), name='activate'),

    #path('anomaly/', views.AnomalyView.as_view(), name="AnomalyView"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

urlpatterns += [
    path('accounts/', include('django.contrib.auth.urls')),
]

# adds the following urls:
#
# accounts/ login/ [name='login']
# accounts/ logout/ [name='logout']
# accounts/ password_change/ [name='password_change']
# accounts/ password_change/done/ [name='password_change_done']
# accounts/ password_reset/ [name='password_reset']
# accounts/ password_reset/done/ [name='password_reset_done']
# accounts/ reset/<uidb64>/<token>/ [name='password_reset_confirm']
# accounts/ reset/done/ [name='password_reset_complete']

# django simple captcha

urlpatterns += [
    path('captcha/', include('captcha.urls')),
]