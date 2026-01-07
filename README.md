# Nexa Auth - AI-Based Mental Health Assessment System

A comprehensive Django-based mental health assessment platform featuring AI-powered multimodal analysis for text and facial emotion detection, designed to support doctors and patients in mental health evaluation and monitoring.

## Overview

Nexa Auth is an advanced healthcare authentication and assessment system that integrates artificial intelligence to provide multimodal mental health evaluations. The system supports separate user roles for doctors and patients, with AI-driven features for conversational analysis, emotion detection from text, and facial emotion recognition.

## Key Features

### AI-Based Mental Health Assessment System
- **Conversational AI**: Integrated chatbot for mental health discussions using fine-tuned language models
- **Emotion Analysis**: Real-time emotion detection from user interactions
- **Mental Health Quizzes**: Structured assessments for mood and mental health evaluation
- **Chat Evaluation**: AI-powered analysis of conversation patterns and emotional states

### Multimodal Features
- **Text Analysis**: Natural language processing for emotion detection in chat conversations
- **Facial Emotion Analysis**: Computer vision-based facial expression recognition using webcam input
- **Combined Assessment**: Integration of text and facial data for comprehensive mental health evaluation

### User Management
- **Role-Based Access**: Separate authentication flows for doctors and patients
- **Doctor Dashboard**: Patient management, assessment reviews, and medical records
- **Patient Dashboard**: Self-assessment tools, doctor directory, and appointment booking
- **Admin Interface**: Comprehensive user and system management

## Tools and Techniques Used

### AI/ML Frameworks
- **Transformers**: Hugging Face transformers for natural language processing and emotion classification
- **PyTorch**: Deep learning framework for model training and inference
- **TensorFlow**: Additional ML framework for computer vision tasks
- **Scikit-learn**: Machine learning utilities for data preprocessing and evaluation

### Computer Vision Libraries
- **OpenCV**: Computer vision operations for image processing
- **FER (Facial Emotion Recognition)**: Specialized library for emotion detection from facial images
- **MediaPipe**: Google's framework for facial landmark detection and analysis

### Backend Technologies
- **Django**: Web framework for robust backend development
- **Django REST Framework**: API development for AI integrations
- **OpenAI API**: Integration with GPT models for advanced conversational AI

### Data Processing
- **Pandas/Numpy**: Data manipulation and analysis
- **SQLite/PostgreSQL**: Database management for user data and assessments

## Deliverables

### Core Application
- Complete Django web application with authentication system
- Role-based dashboards for doctors and patients
- Mental health assessment forms and quizzes
- AI chat interface for patient interactions

### AI Models
- **Conversational AI Model**: Fine-tuned language model for mental health conversations (stored in `conversational_ai_model/`)
- **Emotion Detection Model**: Trained model for text-based emotion analysis (stored in `mental_health_emotion_model/`)
- **Training Scripts**: Python scripts for model training (`train_model.py`, `train_emotion_model.py`)

### Data Assets
- **Emotion Datasets**: Combined datasets for emotion training (`data/mental_health_emotions_combined.csv`)
- **Sample Data**: GoEmotions and other emotion classification datasets
- **Model Checkpoints**: Saved model states for deployment

### Documentation and Setup
- Comprehensive README with setup instructions
- Requirements file with all dependencies
- Database migrations for system initialization

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd nexa_auth
```

### 2. Install Dependencies
Install all required Python packages:
```bash
pip install -r requirements.txt
```

**Note**: Some packages may require additional system dependencies:
- OpenCV may need system libraries for image processing
- PyTorch installation varies by system (CPU/GPU)

### 3. Run Database Migrations
Initialize the database with the required tables:
```bash
python manage.py makemigrations
python manage.py migrate
```

### 4. Create Superuser (Optional)
Create an admin user for accessing the Django admin interface:
```bash
python manage.py createsuperuser
```

### 5. Load AI Models (Optional)
If using pre-trained models, ensure model files are in place:
- `conversational_ai_model/` directory contains the chat model
- `mental_health_emotion_model/` directory contains the emotion detection model

### 6. Run Development Server
Start the Django development server:
```bash
python manage.py runserver
```

The application will be available at `http://127.0.0.1:8000/`

## Application URLs

- **Home**: `http://127.0.0.1:8000/`
- **Login**: `http://127.0.0.1:8000/login/`
- **Registration**: `http://127.0.0.1:8000/register/`
- **Doctor Dashboard**: `http://127.0.0.1:8000/doctor-dashboard/`
- **Patient Dashboard**: `http://127.0.0.1:8000/patient-dashboard/`
- **AI Chat**: `http://127.0.0.1:8000/ai-chat/`
- **Mental Health Quiz**: `http://127.0.0.1:8000/mental-health-quiz/`
- **Admin Interface**: `http://127.0.0.1:8000/admin/`

## Project Structure

```
nexa_auth/
├── manage.py                          # Django management script
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── db.sqlite3                         # SQLite database
├── nexa_auth/                         # Main project settings
│   ├── settings.py                    # Django configuration
│   ├── urls.py                        # URL routing
│   └── ...
├── accounts/                          # Main application
│   ├── models.py                      # User and assessment models
│   ├── views.py                       # Business logic
│   ├── ai_utils.py                    # AI integration utilities
│   ├── forms.py                       # User forms
│   └── migrations/                    # Database migrations
├── conversational_ai_model/           # Chat AI model files
├── mental_health_emotion_model/       # Emotion detection model
├── data/                              # Training datasets
├── templates/                         # HTML templates
├── static/                            # Static assets
└── logs/                              # Training logs
```

## AI Model Training

### Training Conversational AI
```bash
python train_model.py
```

### Training Emotion Detection Model
```bash
python train_emotion_model.py
```

### Downloading Pre-trained Models
```bash
python download_model.py
```

## Deployment

For production deployment:

1. Configure production settings in `nexa_auth/settings.py`:
   - Set `DEBUG = False`
   - Configure `ALLOWED_HOSTS`
   - Set up production database (PostgreSQL recommended)
   - Configure static file serving

2. Collect static files:
   ```bash
   python manage.py collectstatic
   ```

3. Use a production WSGI server:
   ```bash
   gunicorn nexa_auth.wsgi
   ```

## Technology Stack

- **Backend**: Django 5.2.8, Django REST Framework
- **AI/ML**: PyTorch, Transformers, TensorFlow, Scikit-learn
- **Computer Vision**: OpenCV, FER, MediaPipe
- **Frontend**: HTML5, CSS3, Bootstrap 5
- **Database**: SQLite (development), PostgreSQL (production)
- **APIs**: OpenAI API integration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test AI models and web functionality
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For questions or issues, please open an issue in the project repository.

---

**Nexa Auth** - AI-Powered Mental Health Assessment Platform 🤖💙
