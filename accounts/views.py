from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.utils import timezone
from .models import Doctor, Patient, MoodEntry, ChatLog, Conversation, MentalHealthQuiz, Appointment, FacialEmotion
from .forms import UserRegistrationForm, DoctorRegistrationForm, PatientRegistrationForm, PatientUpdateForm, UserLoginForm, MoodQuizForm, AIChatForm, MentalHealthQuizForm
from .serializers import (
    RegisterDoctorSerializer, RegisterPatientSerializer, LoginSerializer,
    DoctorSerializer, PatientSerializer, ChatLogSerializer, AppointmentSerializer
)
from .ai_utils import analyze_message, recommend_doctor, facial_emotion_detector
from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken
import json

def home(request):
    return render(request, 'accounts/home.html')

def about(request):
    return render(request, 'accounts/about.html')

def features(request):
    return render(request, 'accounts/features.html')

def pricing(request):
    return render(request, 'accounts/pricing.html')

def contact(request):
    return render(request, 'accounts/contact.html')

def privacy(request):
    return render(request, 'accounts/privacy.html')

def terms(request):
    return render(request, 'accounts/terms.html')

def register_choice(request):
    """Page to choose between doctor and patient registration"""
    return render(request, 'accounts/register_choice.html')

def register_doctor(request):
    if request.method == 'POST':
        form = DoctorRegistrationForm(request.POST, request.FILES)
        if form.is_valid():
            user = form.save()
            user.set_password(form.cleaned_data['password'])
            user.role = 'doctor'
            user.save()

            # Handle specialization - use other_specialization if "Other" is selected
            specialization = form.cleaned_data['specialization']
            if specialization == 'Other':
                specialization = form.cleaned_data.get('other_specialization', 'Other')

            # Create doctor profile
            Doctor.objects.create(
                user=user,
                full_name=form.cleaned_data['full_name'],
                specialization=specialization,
                experience=form.cleaned_data['experience'],
                clinic_location=form.cleaned_data['clinic_location'],
                available_hours=form.cleaned_data['available_hours'],
                whatsapp_number=form.cleaned_data.get('whatsapp_number'),
                show_whatsapp=form.cleaned_data.get('show_whatsapp', False),
                profile_photo=form.cleaned_data.get('profile_photo'),
                status='pending',
                approved_by_admin=False
            )

            login(request, user)
            messages.success(request, 'Doctor profile created successfully! Your account is pending verification.')
            return redirect('doctor_dashboard')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = DoctorRegistrationForm()
    return render(request, 'accounts/register_doctor.html', {'form': form})

def register_patient(request):
    if request.method == 'POST':
        form = PatientRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.role = 'patient'
            user.save()

            # Create patient profile
            Patient.objects.create(
                user=user,
                date_of_birth=form.cleaned_data.get('date_of_birth'),
                age=form.cleaned_data['age'],
                gender=form.cleaned_data['gender'],
                occupation=form.cleaned_data['occupation'],
                phone=form.cleaned_data.get('phone'),
                address=form.cleaned_data.get('address'),
                city=form.cleaned_data.get('city'),
                state=form.cleaned_data.get('state'),
                emergency_contact=form.cleaned_data.get('emergency_contact'),
                emergency_phone=form.cleaned_data.get('emergency_phone')
            )

            login(request, user)
            messages.success(request, 'Patient account created successfully!')
            return redirect('patient_dashboard')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = PatientRegistrationForm()
    return render(request, 'accounts/register_patient.html', {'form': form})

@login_required
def doctor_dashboard(request):
    if request.user.role != 'doctor':
        messages.error(request, 'Access denied. Doctor account required.')
        return redirect('patient_dashboard')

    # Get doctor's patients (those with appointments)
    doctor = Doctor.objects.get(user=request.user)
    appointments = Appointment.objects.filter(doctor=doctor).select_related('patient__user')
    patient_ids = appointments.values_list('patient', flat=True).distinct()
    patients = Patient.objects.filter(id__in=patient_ids).select_related('user')

    # Analytics data
    analytics_data = {
        'total_patients': patients.count(),
        'total_appointments': appointments.count(),
        'recent_appointments': appointments.order_by('-date')[:5],
    }

    # Prepare chart data for multimodal analytics
    chart_data = prepare_multimodal_analytics(patients)

    context = {
        'patients': patients,
        'analytics_data': analytics_data,
        'chart_data': chart_data,
    }

    return render(request, 'accounts/doctor_dashboard.html', context)

@login_required
def patient_dashboard(request):
    if request.user.role != 'patient':
        messages.error(request, 'Access denied. Patient account required.')
        return redirect('doctor_dashboard')

    # Get recent mood entries
    recent_moods = MoodEntry.objects.filter(user=request.user)[:5]

    # Get recent quiz results
    recent_quiz = MentalHealthQuiz.objects.filter(user=request.user).first()

    # Get recent AI conversations
    recent_conversations = Conversation.objects.filter(user=request.user).order_by('-started_at')[:3]
    for conv in recent_conversations:
        if conv.final_evaluation:
            try:
                conv.parsed_summary = json.loads(conv.final_evaluation)
            except json.JSONDecodeError:
                conv.parsed_summary = None

    # Get recommended doctors (approved doctors)
    recommended_doctors = Doctor.objects.filter(
        approved_by_admin=True
    )[:6]

    context = {
        'recent_moods': recent_moods,
        'recent_quiz': recent_quiz,
        'recent_conversations': recent_conversations,
        'recommended_doctors': recommended_doctors,
    }

    return render(request, 'accounts/patient_dashboard.html', context)

@login_required
def admin_dashboard(request):
    if request.user.role != 'admin':
        messages.error(request, 'Access denied. Admin account required.')
        return redirect('dashboard')

    # Get admin statistics
    total_users = User.objects.count()
    total_doctors = Doctor.objects.count()
    total_patients = Patient.objects.count()
    pending_doctors = Doctor.objects.filter(status='pending').count()
    approved_doctors = Doctor.objects.filter(approved_by_admin=True).count()
    total_quizzes = MentalHealthQuiz.objects.count()
    total_conversations = Conversation.objects.count()

    context = {
        'total_users': total_users,
        'total_doctors': total_doctors,
        'total_patients': total_patients,
        'pending_doctors': pending_doctors,
        'approved_doctors': approved_doctors,
        'total_quizzes': total_quizzes,
        'total_conversations': total_conversations,
    }

    return render(request, 'accounts/admin_dashboard.html', context)

@login_required
def dashboard(request):
    """Generic dashboard that redirects to appropriate user role dashboard"""
    if request.user.role == 'doctor':
        return redirect('doctor_dashboard')
    elif request.user.role == 'patient':
        return redirect('patient_dashboard')
    elif request.user.role == 'admin':
        return redirect('admin_dashboard')
    else:
        messages.error(request, 'Invalid user role. Please contact support.')
        return redirect('login')

@login_required
def assessment_choice(request):
    if request.user.role != 'patient':
        return redirect('dashboard')

    return render(request, 'accounts/assessment_choice.html')

@login_required
def mood_quiz(request):
    if request.user.role != 'patient':
        return redirect('dashboard')

    if request.method == 'POST':
        form = MoodQuizForm(request.POST)
        if form.is_valid():
            mood_entry = MoodEntry.objects.create(
                user=request.user,
                mood=form.cleaned_data['mood'],
                emotions=form.cleaned_data.get('emotions', []),
                energy_level=form.cleaned_data.get('energy_level'),
                sleep_quality=form.cleaned_data.get('sleep_quality'),
                social_interaction=form.cleaned_data.get('social_interaction'),
                stress_level=form.cleaned_data.get('stress_level'),
                concentration=form.cleaned_data.get('concentration'),
                physical_health=form.cleaned_data.get('physical_health'),
                notes=form.cleaned_data.get('notes', '')
            )

            # Generate AI advice based on mood
            generate_ai_advice(request.user, mood_entry)

            messages.success(request, 'Mood entry saved! Check your dashboard for personalized advice.')
            return redirect('patient_dashboard')
    else:
        form = MoodQuizForm()

    return render(request, 'accounts/mood_quiz.html', {'form': form})

@login_required
def mental_health_quiz(request):
    if request.user.role != 'patient':
        return redirect('dashboard')

    # Check if user has already taken the quiz recently (within last 30 days)
    recent_quiz = MentalHealthQuiz.objects.filter(
        user=request.user,
        created_at__gte=timezone.now() - timezone.timedelta(days=30)
    ).first()

    if recent_quiz:
        # Show results if quiz was taken recently
        return render(request, 'accounts/quiz_results.html', {
            'quiz': recent_quiz,
            'recent': True
        })

    if request.method == 'POST':
        form = MentalHealthQuizForm(request.POST, request.FILES)
        if form.is_valid():
            # Collect answers
            answers = {}
            total_score = 0
            for i in range(1, 26):  # 25 questions
                answer = form.cleaned_data[f'q{i}']
                answers[f'q{i}'] = answer
                total_score += answer

            # Get facial image if provided
            facial_image = form.cleaned_data.get('facial_image')
            facial_data = None

            if facial_image:
                try:
                    facial_data = facial_emotion_detector.detect_emotions(facial_image)
                    if "error" in facial_data:
                        messages.warning(request, f'Facial analysis failed: {facial_data["error"]}')
                        facial_data = None
                except Exception as e:
                    messages.warning(request, f'Image processing error: {str(e)}')
                    facial_data = None

            # Create quiz instance
            quiz = MentalHealthQuiz.objects.create(
                user=request.user,
                answers=answers,
                total_score=total_score,
                facial_emotion_data=facial_data
            )

            # Calculate levels and recommendations
            quiz.calculate_levels()
            quiz.save()

            messages.success(request, 'Quiz completed! Here are your results and recommendations.')
            return redirect('quiz_results', quiz_id=quiz.id)
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = MentalHealthQuizForm()

    return render(request, 'accounts/mental_health_quiz.html', {
        'form': form,
        'questions': MentalHealthQuiz.QUIZ_QUESTIONS
    })

@login_required
def quiz_results(request, quiz_id):
    if request.user.role != 'patient':
        return redirect('dashboard')

    try:
        quiz = MentalHealthQuiz.objects.get(id=quiz_id, user=request.user)
    except MentalHealthQuiz.DoesNotExist:
        messages.error(request, 'Quiz not found.')
        return redirect('mental_health_quiz')

    return render(request, 'accounts/quiz_results.html', {
        'quiz': quiz
    })

@login_required
def quiz_history(request):
    if request.user.role != 'patient':
        return redirect('dashboard')

    quizzes = MentalHealthQuiz.objects.filter(user=request.user).order_by('-created_at')

    return render(request, 'accounts/quiz_history.html', {
        'quizzes': quizzes
    })

@login_required
def ai_chat(request):
    if request.user.role != 'patient':
        return redirect('dashboard')

    # Get active conversation or create new one
    conversation = Conversation.objects.filter(user=request.user, ended_at__isnull=True).first()
    if not conversation:
        conversation = Conversation.objects.create(user=request.user)

    conversations = ChatLog.objects.filter(conversation=conversation)[:10]

    # Parse responses for display
    for conv in conversations:
        try:
            conv.parsed_response = json.loads(conv.ai_response)
        except json.JSONDecodeError:
            conv.parsed_response = None

    if request.method == 'POST':
        form = AIChatForm(request.POST, request.FILES)
        if form.is_valid():
            message = form.cleaned_data['message']
            facial_image = form.cleaned_data.get('facial_image')

            # Analyze message with AI (multimodal if image provided)
            emotion, severity_score, ai_response, facial_data = analyze_message(message, facial_image)

            # Create chat log
            chat_log = ChatLog.objects.create(
                conversation=conversation,
                user=request.user,
                message=message,
                emotion=emotion,
                severity_score=severity_score,
                ai_response=ai_response,
                multimodal_prediction=emotion if facial_data else None
            )

            # Store facial emotion data if available
            if facial_data and "error" not in facial_data:
                chat_log.facial_emotion_data = facial_data
                chat_log.save()

                # Create FacialEmotion record
                FacialEmotion.objects.create(
                    chat_log=chat_log,
                    image=facial_image,
                    emotions=facial_data.get('all_emotions', {}),
                    primary_emotion=facial_data.get('primary_emotion'),
                    confidence=facial_data.get('confidence')
                )

            # If severity is high, recommend doctor
            if severity_score > 0.7:
                recommendation = recommend_doctor(emotion)
                # In real implementation, create recommendation record
                print(f"Doctor recommendation: {recommendation}")

            return redirect('ai_chat')
        else:
            # Handle form errors, especially image processing errors
            messages.error(request, 'Please check your input. Image processing may have failed.')
    else:
        form = AIChatForm()

    return render(request, 'accounts/ai_chat.html', {
        'form': form,
        'conversations': conversations
    })

@login_required
def end_chat(request):
    if request.user.role != 'patient':
        return redirect('dashboard')

    # Get active conversation
    conversation = Conversation.objects.filter(user=request.user, ended_at__isnull=True).first()
    if not conversation:
        messages.error(request, 'No active conversation found.')
        return redirect('patient_dashboard')

    # End the conversation
    conversation.ended_at = timezone.now()

    # Get the latest quiz
    quiz = MentalHealthQuiz.objects.filter(user=request.user).order_by('-created_at').first()
    if not quiz:
        messages.error(request, 'No quiz found.')
        return redirect('patient_dashboard')

    # Calculate evaluation
    evaluation = {}
    if quiz.total_score > 45:
        evaluation['needs_doctor'] = True
        evaluation['message'] = "Based on your quiz score and our conversation, we recommend consulting a mental health professional."
        # Recommend approved doctors
        recommended_doctors = Doctor.objects.filter(approved_by_admin=True)[:3]
        evaluation['recommended_doctors'] = [{'name': doc.full_name, 'specialization': doc.specialization, 'id': doc.id} for doc in recommended_doctors]
    else:
        evaluation['needs_doctor'] = False
        evaluation['message'] = "Based on your quiz score and our conversation, here are some self-care solutions you can try."
        evaluation['solutions'] = [
            "Practice daily mindfulness or meditation for 10-15 minutes.",
            "Maintain a regular sleep schedule and healthy eating habits.",
            "Engage in regular physical activity like walking or yoga.",
            "Connect with friends and family for social support.",
            "Consider journaling your thoughts and feelings.",
            "If symptoms persist, consider professional help."
        ]

    conversation.final_evaluation = json.dumps(evaluation)
    conversation.save()

    return render(request, 'accounts/chat_evaluation.html', {
        'evaluation': evaluation,
        'quiz': quiz
    })

@login_required
def doctor_directory(request):
    if request.user.role != 'patient':
        return redirect('dashboard')

    approved_doctors = Doctor.objects.filter(
        approved_by_admin=True
    )

    return render(request, 'accounts/doctor_directory.html', {
        'doctors': approved_doctors
    })


def prepare_multimodal_analytics(patients):
    """Prepare multimodal analytics data for charts"""
    import json
    from collections import defaultdict
    from datetime import datetime, timedelta

    # Initialize data structures
    emotion_trends = defaultdict(list)
    quiz_scores_over_time = {'depression': [], 'anxiety': [], 'dates': []}
    facial_emotions = defaultdict(int)
    text_emotions = defaultdict(int)
    patient_progress = []

    # Get date range (last 30 days)
    end_date = timezone.now()
    start_date = end_date - timedelta(days=30)

    for patient in patients:
        user = patient.user

        # Get quiz results over time
        quizzes = MentalHealthQuiz.objects.filter(
            user=user,
            created_at__gte=start_date
        ).order_by('created_at')

        for quiz in quizzes:
            quiz_scores_over_time['dates'].append(quiz.created_at.strftime('%Y-%m-%d'))
            quiz_scores_over_time['depression'].append({
                'level': quiz.depression_level,
                'score': quiz.total_score
            })
            quiz_scores_over_time['anxiety'].append({
                'level': quiz.anxiety_level,
                'score': quiz.total_score
            })

        # Get chat logs with emotions
        chat_logs = ChatLog.objects.filter(
            user=user,
            timestamp__gte=start_date
        ).select_related('facial_emotion')

        for chat in chat_logs:
            # Text emotions
            text_emotions[chat.emotion] += 1

            # Facial emotions
            if hasattr(chat, 'facial_emotion') and chat.facial_emotion:
                primary = chat.facial_emotion.primary_emotion
                if primary:
                    facial_emotions[primary] += 1

        # Get mood entries
        mood_entries = MoodEntry.objects.filter(
            user=user,
            created_at__gte=start_date
        ).order_by('created_at')

        for mood in mood_entries:
            emotion_trends[mood.created_at.strftime('%Y-%m-%d')].append({
                'mood': mood.mood,
                'emotions': mood.emotions
            })

        # Patient progress summary
        latest_quiz = quizzes.last()
        if latest_quiz:
            progress = {
                'patient_name': f"{user.first_name} {user.last_name}",
                'depression_level': latest_quiz.depression_level,
                'anxiety_level': latest_quiz.anxiety_level,
                'facial_data': bool(latest_quiz.facial_emotion_data),
                'last_assessment': latest_quiz.created_at.strftime('%Y-%m-%d'),
                'recommendations': latest_quiz.recommendations.split('\n')[:3]  # First 3 recommendations
            }
            patient_progress.append(progress)

    # Prepare chart data
    chart_data = {
        'emotion_distribution': {
            'text_emotions': dict(text_emotions),
            'facial_emotions': dict(facial_emotions)
        },
        'quiz_trends': quiz_scores_over_time,
        'emotion_timeline': dict(emotion_trends),
        'patient_progress': patient_progress
    }

    return json.dumps(chart_data)


def generate_ai_advice(user, mood_entry):
    """Generate personalized AI advice based on mood"""
    advice_text = ""

    if mood_entry.mood == 'terrible':
        advice_text = "I'm concerned about how you're feeling. Please consider reaching out to a mental health professional immediately. You're not alone in this."
    elif mood_entry.mood == 'poor':
        advice_text = "It sounds like you're going through a difficult time. Consider these self-care strategies: get adequate sleep, eat nourishing foods, and engage in light physical activity."
    elif mood_entry.mood == 'neutral':
        advice_text = "You're in a neutral space - this can be a good time for reflection and planning. Consider what activities bring you joy and make time for them."
    elif mood_entry.mood == 'good':
        advice_text = "Great to hear you're feeling good! Keep nurturing the positive aspects of your life and consider how you can maintain this positive momentum."
    else:
        advice_text = "Wonderful that you're feeling excellent! This is a great time to reflect on what contributes to your well-being and share your positive energy with others."

    # Recommend doctors based on mood and emotions
    recommended_doctors = Doctor.objects.filter(
        approved_by_admin=True
    )[:3]  # Get top 3 approved doctors

    # For now, just print advice, since AIAdvice model is removed
    print(f"AI Advice for {user.username}: {advice_text}")
    print(f"Recommended doctors: {[doc.full_name for doc in recommended_doctors]}")

def user_login(request):
    if request.method == 'POST':
        form = UserLoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                # Create new conversation for patients on login
                if user.role == 'patient':
                    Conversation.objects.create(user=user)
                messages.success(request, f'Welcome back, {user.username}!')
                return redirect('dashboard')
            else:
                messages.error(request, 'Invalid username or password.')
    else:
        form = UserLoginForm()
    return render(request, 'accounts/login.html', {'form': form})

# API Views

class RegisterDoctorAPIView(generics.CreateAPIView):
    serializer_class = RegisterDoctorSerializer

class RegisterPatientAPIView(generics.CreateAPIView):
    serializer_class = RegisterPatientSerializer

class LoginAPIView(APIView):
    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.validated_data
            refresh = RefreshToken.for_user(user)
            return Response({
                'refresh': str(refresh),
                'access': str(refresh.access_token),
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'role': user.role
                }
            })
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class DoctorListAPIView(generics.ListAPIView):
    queryset = Doctor.objects.filter(approved_by_admin=True)
    serializer_class = DoctorSerializer

class ChatLogListAPIView(generics.ListCreateAPIView):
    serializer_class = ChatLogSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return ChatLog.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        message = serializer.validated_data['message']
        facial_image = serializer.validated_data.get('facial_image')

        emotion, severity_score, ai_response, facial_data = analyze_message(message, facial_image)
        chat_log = serializer.save(
            user=self.request.user,
            emotion=emotion,
            severity_score=severity_score,
            ai_response=ai_response,
            multimodal_prediction=emotion if facial_data else None
        )

        # Store facial emotion data if available
        if facial_data and "error" not in facial_data:
            chat_log.facial_emotion_data = facial_data
            chat_log.save()

            # Create FacialEmotion record
            FacialEmotion.objects.create(
                chat_log=chat_log,
                image=facial_image,
                emotions=facial_data.get('all_emotions', {}),
                primary_emotion=facial_data.get('primary_emotion'),
                confidence=facial_data.get('confidence')
            )

class AppointmentListCreateAPIView(generics.ListCreateAPIView):
    serializer_class = AppointmentSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        if user.role == 'patient':
            return Appointment.objects.filter(patient__user=user)
        elif user.role == 'doctor':
            return Appointment.objects.filter(doctor__user=user)
        return Appointment.objects.none()

    def perform_create(self, serializer):
        if self.request.user.role == 'patient':
            patient = Patient.objects.get(user=self.request.user)
            serializer.save(patient=patient)
