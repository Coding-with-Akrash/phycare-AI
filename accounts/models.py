from django.db import models
from django.contrib.auth.models import AbstractUser
from django.core.exceptions import ValidationError

class User(AbstractUser):
    ROLE_CHOICES = (
        ('doctor', 'Doctor'),
        ('patient', 'Patient'),
        ('admin', 'Admin'),
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='patient')
    is_verified = models.BooleanField(default=False)

    @property
    def profile(self):
        if self.role == 'patient':
            return getattr(self, 'patient', None)
        elif self.role == 'doctor':
            return getattr(self, 'doctor', None)
        return None

    def __str__(self):
        return f"{self.username} ({self.role})"

class Doctor(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    full_name = models.CharField(max_length=100)
    specialization = models.CharField(max_length=100)
    experience = models.IntegerField()
    clinic_location = models.CharField(max_length=150)
    available_hours = models.CharField(max_length=100)
    whatsapp_number = models.CharField(max_length=15)
    show_whatsapp = models.BooleanField(default=True)
    status = models.CharField(max_length=20, default="pending")
    approved_by_admin = models.BooleanField(default=False)
    profile_photo = models.ImageField(upload_to='doctor_photos/', blank=True, null=True)

    def __str__(self):
        return f"Dr. {self.full_name} - {self.specialization}"

    def get_whatsapp_link(self):
        if self.whatsapp_number and self.show_whatsapp:
            number = ''.join(filter(str.isdigit, self.whatsapp_number))
            return f"https://wa.me/{number}"
        return None

class Patient(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    date_of_birth = models.DateField(null=True, blank=True)
    age = models.IntegerField()
    gender = models.CharField(max_length=10)
    occupation = models.CharField(max_length=100)
    phone = models.CharField(max_length=15, blank=True, null=True)
    address = models.TextField(blank=True, null=True)
    city = models.CharField(max_length=100, blank=True, null=True)
    state = models.CharField(max_length=100, blank=True, null=True)
    emergency_contact = models.CharField(max_length=100, blank=True, null=True)
    emergency_phone = models.CharField(max_length=15, blank=True, null=True)

    def __str__(self):
        return f"{self.user.username} - Patient"

class MoodEntry(models.Model):
    MOOD_CHOICES = [
        ('excellent', 'Excellent'),
        ('good', 'Good'),
        ('neutral', 'Neutral'),
        ('poor', 'Poor'),
        ('terrible', 'Terrible'),
    ]

    ENERGY_CHOICES = [
        ('very_high', 'Very High'),
        ('high', 'High'),
        ('moderate', 'Moderate'),
        ('low', 'Low'),
        ('very_low', 'Very Low'),
    ]

    SLEEP_CHOICES = [
        ('excellent', 'Excellent'),
        ('good', 'Good'),
        ('fair', 'Fair'),
        ('poor', 'Poor'),
        ('terrible', 'Terrible'),
    ]

    SOCIAL_CHOICES = [
        ('very_satisfied', 'Very Satisfied'),
        ('satisfied', 'Satisfied'),
        ('neutral', 'Neutral'),
        ('dissatisfied', 'Dissatisfied'),
        ('very_dissatisfied', 'Very Dissatisfied'),
    ]

    STRESS_CHOICES = [
        ('none', 'None'),
        ('low', 'Low'),
        ('moderate', 'Moderate'),
        ('high', 'High'),
        ('extreme', 'Extreme'),
    ]

    CONCENTRATION_CHOICES = [
        ('excellent', 'Excellent'),
        ('good', 'Good'),
        ('fair', 'Fair'),
        ('poor', 'Poor'),
        ('terrible', 'Terrible'),
    ]

    HEALTH_CHOICES = [
        ('excellent', 'Excellent'),
        ('good', 'Good'),
        ('fair', 'Fair'),
        ('poor', 'Poor'),
        ('terrible', 'Terrible'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='mood_entries')
    mood = models.CharField(max_length=20, choices=MOOD_CHOICES)
    emotions = models.JSONField(default=list)
    energy_level = models.CharField(max_length=20, choices=ENERGY_CHOICES, blank=True, null=True)
    sleep_quality = models.CharField(max_length=20, choices=SLEEP_CHOICES, blank=True, null=True)
    social_interaction = models.CharField(max_length=20, choices=SOCIAL_CHOICES, blank=True, null=True)
    stress_level = models.CharField(max_length=20, choices=STRESS_CHOICES, blank=True, null=True)
    concentration = models.CharField(max_length=20, choices=CONCENTRATION_CHOICES, blank=True, null=True)
    physical_health = models.CharField(max_length=20, choices=HEALTH_CHOICES, blank=True, null=True)
    notes = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

class Conversation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    final_evaluation = models.JSONField(null=True, blank=True)  # Store final evaluation when conversation ends

    class Meta:
        ordering = ['-started_at']

    @property
    def is_active(self):
        return self.ended_at is None

    @property
    def overall_emotion(self):
        if self.final_evaluation:
            return self.final_evaluation.get('emotion', 'N/A')
        return 'N/A'

    @property
    def overall_severity(self):
        if self.final_evaluation:
            return self.final_evaluation.get('severity', 0)
        return 0

    def __str__(self):
        return f"Conversation {self.id} - {self.user.username} ({self.started_at.date()})"

class ChatLog(models.Model):
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # Keep for backward compatibility or remove if not needed
    message = models.TextField()
    emotion = models.CharField(max_length=50)
    severity_score = models.FloatField()
    ai_response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    facial_emotion_data = models.JSONField(blank=True, null=True)
    multimodal_prediction = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        ordering = ['timestamp']

class FacialEmotion(models.Model):
    chat_log = models.OneToOneField(ChatLog, on_delete=models.CASCADE, related_name='facial_emotion')
    image = models.ImageField(upload_to='facial_images/', blank=True, null=True)
    emotions = models.JSONField(default=dict)  # {'happy': 0.8, 'sad': 0.1, ...}
    primary_emotion = models.CharField(max_length=50, blank=True, null=True)
    confidence = models.FloatField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Facial Emotion for ChatLog {self.chat_log.id} - {self.primary_emotion}"

class MentalHealthQuiz(models.Model):
    QUIZ_QUESTIONS = [
        # PHQ-9 Questions (Depression)
        {
            "id": 1,
            "question": "Over the last 2 weeks, how often have you been bothered by feeling down, depressed, or hopeless?",
            "options": [
                {"text": "Not at all", "score": 0},
                {"text": "Several days", "score": 1},
                {"text": "More than half the days", "score": 2},
                {"text": "Nearly every day", "score": 3},
                {"text": "Every day", "score": 4},
                {"text": "All the time", "score": 5}
            ]
        },
        {
            "id": 2,
            "question": "Over the last 2 weeks, how often have you been bothered by little interest or pleasure in doing things?",
            "options": [
                {"text": "Not at all", "score": 0},
                {"text": "Several days", "score": 1},
                {"text": "More than half the days", "score": 2},
                {"text": "Nearly every day", "score": 3},
                {"text": "Every day", "score": 4},
                {"text": "All the time", "score": 5}
            ]
        },
        {
            "id": 3,
            "question": "Over the last 2 weeks, how often have you been bothered by feeling tired or having little energy?",
            "options": [
                {"text": "Not at all", "score": 0},
                {"text": "Several days", "score": 1},
                {"text": "More than half the days", "score": 2},
                {"text": "Nearly every day", "score": 3},
                {"text": "Every day", "score": 4},
                {"text": "All the time", "score": 5}
            ]
        },
        {
            "id": 4,
            "question": "Over the last 2 weeks, how often have you been bothered by poor appetite or overeating?",
            "options": [
                {"text": "Not at all", "score": 0},
                {"text": "Several days", "score": 1},
                {"text": "More than half the days", "score": 2},
                {"text": "Nearly every day", "score": 3},
                {"text": "Every day", "score": 4},
                {"text": "All the time", "score": 5}
            ]
        },
        {
            "id": 5,
            "question": "Over the last 2 weeks, how often have you been bothered by feeling bad about yourself or that you are a failure?",
            "options": [
                {"text": "Not at all", "score": 0},
                {"text": "Several days", "score": 1},
                {"text": "More than half the days", "score": 2},
                {"text": "Nearly every day", "score": 3},
                {"text": "Every day", "score": 4},
                {"text": "All the time", "score": 5}
            ]
        },
        {
            "id": 6,
            "question": "Over the last 2 weeks, how often have you been bothered by trouble concentrating on things?",
            "options": [
                {"text": "Not at all", "score": 0},
                {"text": "Several days", "score": 1},
                {"text": "More than half the days", "score": 2},
                {"text": "Nearly every day", "score": 3},
                {"text": "Every day", "score": 4},
                {"text": "All the time", "score": 5}
            ]
        },
        {
            "id": 7,
            "question": "Over the last 2 weeks, how often have you been bothered by moving or speaking so slowly that other people could have noticed?",
            "options": [
                {"text": "Not at all", "score": 0},
                {"text": "Several days", "score": 1},
                {"text": "More than half the days", "score": 2},
                {"text": "Nearly every day", "score": 3},
                {"text": "Every day", "score": 4},
                {"text": "All the time", "score": 5}
            ]
        },
        {
            "id": 8,
            "question": "Over the last 2 weeks, how often have you been bothered by thoughts that you would be better off dead or of hurting yourself?",
            "options": [
                {"text": "Not at all", "score": 0},
                {"text": "Several days", "score": 1},
                {"text": "More than half the days", "score": 2},
                {"text": "Nearly every day", "score": 3},
                {"text": "Every day", "score": 4},
                {"text": "All the time", "score": 5}
            ]
        },
        # GAD-7 Questions (Anxiety)
        {
            "id": 9,
            "question": "Over the last 2 weeks, how often have you been bothered by feeling nervous, anxious, or on edge?",
            "options": [
                {"text": "Not at all", "score": 0},
                {"text": "Several days", "score": 1},
                {"text": "More than half the days", "score": 2},
                {"text": "Nearly every day", "score": 3},
                {"text": "Every day", "score": 4},
                {"text": "All the time", "score": 5}
            ]
        },
        {
            "id": 10,
            "question": "Over the last 2 weeks, how often have you been bothered by not being able to stop or control worrying?",
            "options": [
                {"text": "Not at all", "score": 0},
                {"text": "Several days", "score": 1},
                {"text": "More than half the days", "score": 2},
                {"text": "Nearly every day", "score": 3},
                {"text": "Every day", "score": 4},
                {"text": "All the time", "score": 5}
            ]
        },
        {
            "id": 11,
            "question": "Over the last 2 weeks, how often have you been bothered by worrying too much about different things?",
            "options": [
                {"text": "Not at all", "score": 0},
                {"text": "Several days", "score": 1},
                {"text": "More than half the days", "score": 2},
                {"text": "Nearly every day", "score": 3},
                {"text": "Every day", "score": 4},
                {"text": "All the time", "score": 5}
            ]
        },
        {
            "id": 12,
            "question": "Over the last 2 weeks, how often have you been bothered by trouble relaxing?",
            "options": [
                {"text": "Not at all", "score": 0},
                {"text": "Several days", "score": 1},
                {"text": "More than half the days", "score": 2},
                {"text": "Nearly every day", "score": 3},
                {"text": "Every day", "score": 4},
                {"text": "All the time", "score": 5}
            ]
        },
        {
            "id": 13,
            "question": "Over the last 2 weeks, how often have you been bothered by being so restless that it's hard to sit still?",
            "options": [
                {"text": "Not at all", "score": 0},
                {"text": "Several days", "score": 1},
                {"text": "More than half the days", "score": 2},
                {"text": "Nearly every day", "score": 3},
                {"text": "Every day", "score": 4},
                {"text": "All the time", "score": 5}
            ]
        },
        {
            "id": 14,
            "question": "Over the last 2 weeks, how often have you been bothered by becoming easily annoyed or irritable?",
            "options": [
                {"text": "Not at all", "score": 0},
                {"text": "Several days", "score": 1},
                {"text": "More than half the days", "score": 2},
                {"text": "Nearly every day", "score": 3},
                {"text": "Every day", "score": 4},
                {"text": "All the time", "score": 5}
            ]
        },
        {
            "id": 15,
            "question": "Over the last 2 weeks, how often have you been bothered by feeling afraid as if something awful might happen?",
            "options": [
                {"text": "Not at all", "score": 0},
                {"text": "Several days", "score": 1},
                {"text": "More than half the days", "score": 2},
                {"text": "Nearly every day", "score": 3},
                {"text": "Every day", "score": 4},
                {"text": "All the time", "score": 5}
            ]
        },
        # Additional Mental Health Questions
        {
            "id": 16,
            "question": "How often do you feel overwhelmed by your responsibilities?",
            "options": [
                {"text": "Never", "score": 0},
                {"text": "Rarely", "score": 1},
                {"text": "Sometimes", "score": 2},
                {"text": "Often", "score": 3},
                {"text": "Very often", "score": 4},
                {"text": "Always", "score": 5}
            ]
        },
        {
            "id": 17,
            "question": "How often do you have trouble sleeping due to stress or worry?",
            "options": [
                {"text": "Never", "score": 0},
                {"text": "Rarely", "score": 1},
                {"text": "Sometimes", "score": 2},
                {"text": "Often", "score": 3},
                {"text": "Very often", "score": 4},
                {"text": "Always", "score": 5}
            ]
        },
        {
            "id": 18,
            "question": "How often do you feel isolated or lonely?",
            "options": [
                {"text": "Never", "score": 0},
                {"text": "Rarely", "score": 1},
                {"text": "Sometimes", "score": 2},
                {"text": "Often", "score": 3},
                {"text": "Very often", "score": 4},
                {"text": "Always", "score": 5}
            ]
        },
        {
            "id": 19,
            "question": "How often do you experience mood swings?",
            "options": [
                {"text": "Never", "score": 0},
                {"text": "Rarely", "score": 1},
                {"text": "Sometimes", "score": 2},
                {"text": "Often", "score": 3},
                {"text": "Very often", "score": 4},
                {"text": "Always", "score": 5}
            ]
        },
        {
            "id": 20,
            "question": "How often do you feel confident in handling life's challenges?",
            "options": [
                {"text": "Always", "score": 0},
                {"text": "Very often", "score": 1},
                {"text": "Often", "score": 2},
                {"text": "Sometimes", "score": 3},
                {"text": "Rarely", "score": 4},
                {"text": "Never", "score": 5}
            ]
        },
        {
            "id": 21,
            "question": "How often do you engage in activities that bring you joy?",
            "options": [
                {"text": "Always", "score": 0},
                {"text": "Very often", "score": 1},
                {"text": "Often", "score": 2},
                {"text": "Sometimes", "score": 3},
                {"text": "Rarely", "score": 4},
                {"text": "Never", "score": 5}
            ]
        },
        {
            "id": 22,
            "question": "How often do you feel supported by your friends and family?",
            "options": [
                {"text": "Always", "score": 0},
                {"text": "Very often", "score": 1},
                {"text": "Often", "score": 2},
                {"text": "Sometimes", "score": 3},
                {"text": "Rarely", "score": 4},
                {"text": "Never", "score": 5}
            ]
        },
        {
            "id": 23,
            "question": "How often do you experience physical symptoms due to stress (headaches, stomach issues)?",
            "options": [
                {"text": "Never", "score": 0},
                {"text": "Rarely", "score": 1},
                {"text": "Sometimes", "score": 2},
                {"text": "Often", "score": 3},
                {"text": "Very often", "score": 4},
                {"text": "Always", "score": 5}
            ]
        },
        {
            "id": 24,
            "question": "How often do you find it difficult to express your emotions?",
            "options": [
                {"text": "Never", "score": 0},
                {"text": "Rarely", "score": 1},
                {"text": "Sometimes", "score": 2},
                {"text": "Often", "score": 3},
                {"text": "Very often", "score": 4},
                {"text": "Always", "score": 5}
            ]
        },
        {
            "id": 25,
            "question": "How often do you feel motivated to achieve your goals?",
            "options": [
                {"text": "Always", "score": 0},
                {"text": "Very often", "score": 1},
                {"text": "Often", "score": 2},
                {"text": "Sometimes", "score": 3},
                {"text": "Rarely", "score": 4},
                {"text": "Never", "score": 5}
            ]
        }
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    answers = models.JSONField()  # Store answers as JSON
    total_score = models.IntegerField()
    depression_level = models.CharField(max_length=20, choices=[
        ('minimal', 'Minimal'),
        ('mild', 'Mild'),
        ('moderate', 'Moderate'),
        ('moderately_severe', 'Moderately Severe'),
        ('severe', 'Severe')
    ])
    anxiety_level = models.CharField(max_length=20, choices=[
        ('minimal', 'Minimal'),
        ('mild', 'Mild'),
        ('moderate', 'Moderate'),
        ('moderately_severe', 'Moderately Severe'),
        ('severe', 'Severe')
    ])
    recommendations = models.TextField()
    facial_emotion_data = models.JSONField(blank=True, null=True)  # Store facial emotion analysis if image provided
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Mental Health Quiz - {self.user.username} ({self.created_at.date()})"

    def calculate_levels(self):
        """Calculate depression and anxiety levels based on expanded mental health assessment"""
        # PHQ-9 scoring (questions 1-9: depression items)
        depression_score = sum(self.answers.get(f'q{i}', 0) for i in range(1, 10))

        # GAD-7 scoring (questions 10-16: anxiety items)
        anxiety_score = sum(self.answers.get(f'q{i}', 0) for i in range(10, 17))

        # General mental health score (questions 17-25: additional factors)
        general_score = sum(self.answers.get(f'q{i}', 0) for i in range(17, 26))

        # Adjust scores for the expanded scale (0-5 instead of 0-3)
        # Normalize to original scales for level determination
        normalized_depression = min(depression_score * 3 / 5, 27)  # Max 27 for PHQ-9
        normalized_anxiety = min(anxiety_score * 3 / 5, 21)       # Max 21 for GAD-7

        # Determine depression level
        if normalized_depression <= 4:
            self.depression_level = 'minimal'
        elif normalized_depression <= 9:
            self.depression_level = 'mild'
        elif normalized_depression <= 14:
            self.depression_level = 'moderate'
        elif normalized_depression <= 19:
            self.depression_level = 'moderately_severe'
        else:
            self.depression_level = 'severe'

        # Determine anxiety level
        if normalized_anxiety <= 4:
            self.anxiety_level = 'minimal'
        elif normalized_anxiety <= 9:
            self.anxiety_level = 'mild'
        elif normalized_anxiety <= 14:
            self.anxiety_level = 'moderate'
        elif normalized_anxiety <= 19:
            self.anxiety_level = 'moderately_severe'
        else:
            self.anxiety_level = 'severe'

        # Generate recommendations
        self.generate_recommendations()

    def generate_recommendations(self):
        """Generate personalized recommendations based on scores"""
        recommendations = []

        if self.depression_level in ['moderate', 'moderately_severe', 'severe']:
            recommendations.append("Consider consulting a mental health professional for comprehensive evaluation and treatment.")
            recommendations.append("Cognitive Behavioral Therapy (CBT) may be beneficial for managing depressive symptoms.")

        if self.anxiety_level in ['moderate', 'moderately_severe', 'severe']:
            recommendations.append("Practice relaxation techniques such as deep breathing exercises or progressive muscle relaxation.")
            recommendations.append("Consider speaking with a therapist about anxiety management strategies.")

        if self.depression_level == 'mild' or self.anxiety_level == 'mild':
            recommendations.append("Maintain a consistent sleep schedule and engage in regular physical activity.")
            recommendations.append("Consider mindfulness or meditation practices to help manage symptoms.")

        if self.depression_level == 'minimal' and self.anxiety_level == 'minimal':
            recommendations.append("Continue practicing good self-care habits to maintain your mental wellness.")
            recommendations.append("Regular exercise, healthy eating, and social connections are important for mental health.")

        # General recommendations
        recommendations.extend([
            "Reach out to trusted friends or family for support.",
            "Consider journaling your thoughts and feelings.",
            "Practice stress-reduction techniques like meditation or yoga.",
            "Maintain a healthy sleep schedule and balanced diet."
        ])

        self.recommendations = '\n'.join(recommendations)

class Appointment(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE)
    date = models.DateField()
    time = models.TimeField()
    status = models.CharField(max_length=20, default="pending")

    def __str__(self):
        return f"Appointment: {self.patient} with {self.doctor} on {self.date}"
