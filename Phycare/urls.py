"""
URL configuration for nexa_auth project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views
from accounts import views as accounts_views
from rest_framework_simplejwt.views import TokenRefreshView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', accounts_views.home, name='home'),
    path('register/', accounts_views.register_choice, name='register'),
    path('register/doctor/', accounts_views.register_doctor, name='register_doctor'),
    path('register/patient/', accounts_views.register_patient, name='register_patient'),
    path('login/', accounts_views.user_login, name='login'),
    path('logout/', auth_views.LogoutView.as_view(template_name='accounts/logout.html', http_method_names=['get', 'post']), name='logout'),
    path('dashboard/', accounts_views.dashboard, name='dashboard'),
    path('admin-dashboard/', accounts_views.admin_dashboard, name='admin_dashboard'),
    path('doctor-dashboard/', accounts_views.doctor_dashboard, name='doctor_dashboard'),
    path('patient-dashboard/', accounts_views.patient_dashboard, name='patient_dashboard'),
    path('mood-quiz/', accounts_views.mood_quiz, name='mood_quiz'),
    path('assessment-choice/', accounts_views.assessment_choice, name='assessment_choice'),
    path('mental-health-quiz/', accounts_views.mental_health_quiz, name='mental_health_quiz'),
    path('quiz-results/<int:quiz_id>/', accounts_views.quiz_results, name='quiz_results'),
    path('quiz-history/', accounts_views.quiz_history, name='quiz_history'),
    path('ai-chat/', accounts_views.ai_chat, name='ai_chat'),
    path('end-chat/', accounts_views.end_chat, name='end_chat'),
    path('doctor-directory/', accounts_views.doctor_directory, name='doctor_directory'),
    path('about/', accounts_views.about, name='about'),
    path('features/', accounts_views.features, name='features'),
    path('pricing/', accounts_views.pricing, name='pricing'),
    path('contact/', accounts_views.contact, name='contact'),
    path('privacy/', accounts_views.privacy, name='privacy'),
    path('terms/', accounts_views.terms, name='terms'),

    # API routes
    path('api/register/doctor/', accounts_views.RegisterDoctorAPIView.as_view(), name='api_register_doctor'),
    path('api/register/patient/', accounts_views.RegisterPatientAPIView.as_view(), name='api_register_patient'),
    path('api/login/', accounts_views.LoginAPIView.as_view(), name='api_login'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/doctors/', accounts_views.DoctorListAPIView.as_view(), name='api_doctors'),
    path('api/chat/', accounts_views.ChatLogListAPIView.as_view(), name='api_chat'),
    path('api/appointments/', accounts_views.AppointmentListCreateAPIView.as_view(), name='api_appointments'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
