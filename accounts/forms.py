from django import forms
from django.core.exceptions import ValidationError
from .models import User, Doctor, Patient

class UserRegistrationForm(forms.ModelForm):
    password = forms.CharField(
        label='Password',
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your password'
        })
    )
    confirm_password = forms.CharField(
        label='Confirm Password',
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Confirm your password'
        })
    )
    
    class Meta:
        model = User
        fields = ['username', 'email', 'password']
        widgets = {
            'username': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter your username'
            }),
            'email': forms.EmailInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter your email'
            }),
        }
    
    def clean_confirm_password(self):
        password = self.cleaned_data.get('password')
        confirm_password = self.cleaned_data.get('confirm_password')
        if password != confirm_password:
            raise ValidationError("Passwords don't match")
        return confirm_password
    
    def clean_username(self):
        username = self.cleaned_data.get('username')
        if User.objects.filter(username=username).exists():
            raise ValidationError("Username already exists")
        return username
    
    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise ValidationError("Email already exists")
        return email

class DoctorRegistrationForm(UserRegistrationForm):
    # Doctor specific fields - Only psychological/psychiatric specializations
    SPECIALIZATION_CHOICES = [
        ('', 'Select your specialization'),
        ('Clinical Psychologist', 'Clinical Psychologist'),
        ('Counseling Psychologist', 'Counseling Psychologist'),
        ('Psychiatrist', 'Psychiatrist'),
        ('Child Psychologist', 'Child Psychologist'),
        ('Marriage and Family Therapist', 'Marriage and Family Therapist'),
        ('Addiction Counselor', 'Addiction Counselor'),
        ('Trauma Specialist', 'Trauma Specialist'),
        ('Cognitive Behavioral Therapist', 'Cognitive Behavioral Therapist'),
        ('Mental Health Counselor', 'Mental Health Counselor'),
        ('Other', 'Other (please specify)'),
    ]

    specialization = forms.ChoiceField(
        choices=SPECIALIZATION_CHOICES,
        widget=forms.Select(attrs={
            'class': 'form-control',
            'id': 'doctor_specialization'
        })
    )
    other_specialization = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Please specify your specialization',
            'id': 'doctor_other_specialization',
            'style': 'display: none;'
        })
    )
    experience = forms.IntegerField(
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Years of experience',
            'id': 'doctor_experience',
            'min': '0',
            'max': '50'
        })
    )
    clinic_location = forms.CharField(
        max_length=150,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Clinic/Hospital location',
            'id': 'doctor_clinic'
        })
    )
    available_hours = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., Mon-Fri: 9AM-5PM',
            'id': 'doctor_hours'
        })
    )
    full_name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Your full professional name',
            'id': 'doctor_full_name'
        })
    )
    whatsapp_number = forms.CharField(
        max_length=15,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Your WhatsApp number (optional)',
            'id': 'doctor_whatsapp'
        })
    )
    show_whatsapp = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input',
            'id': 'doctor_show_whatsapp'
        })
    )
    profile_photo = forms.ImageField(
        required=False,
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'id': 'doctor_photo',
            'accept': 'image/*'
        })
    )

    class Meta(UserRegistrationForm.Meta):
        fields = ['username', 'email', 'password', 'confirm_password',
                 'full_name', 'specialization', 'other_specialization', 'experience',
                 'clinic_location', 'available_hours', 'whatsapp_number', 'show_whatsapp', 'profile_photo']

class PatientRegistrationForm(UserRegistrationForm):
    # Patient specific fields
    date_of_birth = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={
            'class': 'form-control',
            'type': 'date',
            'id': 'patient_dob'
        })
    )
    age = forms.IntegerField(
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Your age',
            'id': 'patient_age',
            'min': '1',
            'max': '120'
        })
    )
    gender = forms.CharField(
        max_length=10,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Male/Female/Other',
            'id': 'patient_gender'
        })
    )
    occupation = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Your occupation',
            'id': 'patient_occupation'
        })
    )
    phone = forms.CharField(
        max_length=15,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Your phone number',
            'id': 'patient_phone'
        })
    )
    address = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Your address',
            'id': 'patient_address'
        })
    )
    city = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Your city',
            'id': 'patient_city'
        })
    )
    state = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Your state/province',
            'id': 'patient_state'
        })
    )
    emergency_contact = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Emergency contact name',
            'id': 'patient_emergency_contact'
        })
    )
    emergency_phone = forms.CharField(
        max_length=15,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Emergency contact phone',
            'id': 'patient_emergency_phone'
        })
    )

    class Meta(UserRegistrationForm.Meta):
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'password', 'confirm_password',
                  'date_of_birth', 'age', 'gender', 'occupation', 'phone',
                  'address', 'city', 'state', 'emergency_contact', 'emergency_phone']
        widgets = {
            'first_name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter your first name'
            }),
            'last_name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter your last name'
            }),
        }

class MoodQuizForm(forms.Form):
    mood = forms.ChoiceField(
        choices=[
            ('excellent', '😊 Excellent - Feeling great!'),
            ('good', '🙂 Good - Feeling positive'),
            ('neutral', '😐 Neutral - Feeling okay'),
            ('poor', '😟 Poor - Feeling down'),
            ('terrible', '😢 Terrible - Really struggling'),
        ],
        widget=forms.RadioSelect(attrs={'class': 'mood-radio'}),
        label="How are you feeling today?"
    )

    emotions = forms.MultipleChoiceField(
        choices=[
            ('happy', 'Happy'),
            ('sad', 'Sad'),
            ('anxious', 'Anxious'),
            ('angry', 'Angry'),
            ('stressed', 'Stressed'),
            ('excited', 'Excited'),
            ('tired', 'Tired'),
            ('hopeful', 'Hopeful'),
            ('overwhelmed', 'Overwhelmed'),
            ('peaceful', 'Peaceful'),
            ('lonely', 'Lonely'),
            ('confused', 'Confused'),
            ('frustrated', 'Frustrated'),
            ('motivated', 'Motivated'),
            ('grateful', 'Grateful'),
            ('worried', 'Worried'),
            ('content', 'Content'),
            ('irritable', 'Irritable'),
            ('calm', 'Calm'),
            ('energetic', 'Energetic'),
        ],
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'emotion-checkbox'}),
        required=False,
        label="What emotions are you experiencing? (Select all that apply)"
    )

    energy_level = forms.ChoiceField(
        choices=[
            ('very_high', '🔋 Very High - Full of energy'),
            ('high', '⚡ High - Good energy levels'),
            ('moderate', '🔋 Moderate - Normal energy'),
            ('low', '🪫 Low - Feeling drained'),
            ('very_low', '🔋 Very Low - Completely exhausted'),
        ],
        widget=forms.RadioSelect(attrs={'class': 'energy-radio'}),
        label="How is your energy level today?"
    )

    sleep_quality = forms.ChoiceField(
        choices=[
            ('excellent', '😴 Excellent - Slept very well'),
            ('good', '😊 Good - Slept well'),
            ('fair', '😐 Fair - Slept okay'),
            ('poor', '😟 Poor - Slept badly'),
            ('terrible', '😫 Terrible - Barely slept'),
        ],
        widget=forms.RadioSelect(attrs={'class': 'sleep-radio'}),
        label="How did you sleep last night?"
    )

    social_interaction = forms.ChoiceField(
        choices=[
            ('very_satisfied', '👥 Very Satisfied - Great social connections'),
            ('satisfied', '🙂 Satisfied - Good social life'),
            ('neutral', '😐 Neutral - Average social interaction'),
            ('dissatisfied', '😕 Dissatisfied - Limited social contact'),
            ('very_dissatisfied', '😢 Very Dissatisfied - Feeling isolated'),
        ],
        widget=forms.RadioSelect(attrs={'class': 'social-radio'}),
        label="How satisfied are you with your social interactions?"
    )

    stress_level = forms.ChoiceField(
        choices=[
            ('none', '😌 None - Completely relaxed'),
            ('low', '🙂 Low - Mild stress'),
            ('moderate', '😐 Moderate - Manageable stress'),
            ('high', '😟 High - Significant stress'),
            ('extreme', '😰 Extreme - Overwhelming stress'),
        ],
        widget=forms.RadioSelect(attrs={'class': 'stress-radio'}),
        label="What is your current stress level?"
    )

    concentration = forms.ChoiceField(
        choices=[
            ('excellent', '🎯 Excellent - Very focused'),
            ('good', '✅ Good - Able to concentrate'),
            ('fair', '😐 Fair - Some difficulty focusing'),
            ('poor', '😕 Poor - Hard to concentrate'),
            ('terrible', '❌ Terrible - Cannot focus at all'),
        ],
        widget=forms.RadioSelect(attrs={'class': 'concentration-radio'}),
        label="How is your ability to concentrate today?"
    )

    physical_health = forms.ChoiceField(
        choices=[
            ('excellent', '💪 Excellent - Feeling strong'),
            ('good', '👍 Good - Feeling healthy'),
            ('fair', '😐 Fair - Some physical discomfort'),
            ('poor', '😷 Poor - Physical health issues'),
            ('terrible', '🤒 Terrible - Significant health problems'),
        ],
        widget=forms.RadioSelect(attrs={'class': 'health-radio'}),
        label="How is your physical health today?"
    )

    notes = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 4,
            'placeholder': 'Any additional thoughts, triggers, or notes about your day... (optional)'
        }),
        required=False,
        label="Additional Notes"
    )

class AIChatForm(forms.Form):
    message = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Share what\'s on your mind...',
            'id': 'ai-message-input'
        }),
        label="Your message"
    )
    facial_image = forms.ImageField(
        required=False,
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'id': 'facial-image-input',
            'accept': 'image/*'
        }),
        label="Upload a photo (optional) - For facial emotion analysis"
    )

class PatientUpdateForm(forms.ModelForm):
    # Patient specific fields for updating
    date_of_birth = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={
            'class': 'form-control',
            'type': 'date',
            'id': 'patient_dob_update'
        })
    )
    age = forms.IntegerField(
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Your age',
            'id': 'patient_age_update',
            'min': '1',
            'max': '120'
        })
    )
    gender = forms.CharField(
        max_length=10,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Male/Female/Other',
            'id': 'patient_gender_update'
        })
    )
    occupation = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Your occupation',
            'id': 'patient_occupation_update'
        })
    )
    phone = forms.CharField(
        max_length=15,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Your phone number',
            'id': 'patient_phone_update'
        })
    )
    address = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Your address',
            'id': 'patient_address_update'
        })
    )
    city = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Your city',
            'id': 'patient_city_update'
        })
    )
    state = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Your state/province',
            'id': 'patient_state_update'
        })
    )
    emergency_contact = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Emergency contact name',
            'id': 'patient_emergency_contact_update'
        })
    )
    emergency_phone = forms.CharField(
        max_length=15,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Emergency contact phone',
            'id': 'patient_emergency_phone_update'
        })
    )

    class Meta:
        model = Patient
        fields = ['date_of_birth', 'age', 'gender', 'occupation', 'phone',
                  'address', 'city', 'state', 'emergency_contact', 'emergency_phone']

class UserLoginForm(forms.Form):
    username = forms.CharField(
        label='Username',
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your username'
        })
    )
    password = forms.CharField(
        label='Password',
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your password'
        })
    )

class MentalHealthQuizForm(forms.Form):
    facial_image = forms.ImageField(
        required=False,
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'id': 'quiz-facial-image',
            'accept': 'image/*'
        }),
        label="Upload a photo (optional) - For enhanced facial emotion analysis"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from .models import MentalHealthQuiz

        for question in MentalHealthQuiz.QUIZ_QUESTIONS:
            choices = [(option['score'], option['text']) for option in question['options']]
            self.fields[f'q{question["id"]}'] = forms.ChoiceField(
                label=question['question'],
                choices=choices,
                widget=forms.RadioSelect,
                required=True
            )