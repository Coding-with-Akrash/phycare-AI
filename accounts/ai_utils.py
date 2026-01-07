import json

# Emotion detection using RoBERTa model
# Primary: Fine-tuned mental health model (if available)
# Fallback: cardiffnlp/twitter-roberta-base-emotion (pre-trained on Twitter data)
FINE_TUNED_MODEL_PATH = './mental_health_emotion_model'
FALLBACK_MODEL_NAME = 'cardiffnlp/twitter-roberta-base-emotion'

# Conversational AI model
CONVERSATIONAL_MODEL_PATH = './conversational_ai_model'

# Emotions for our mental health model
MENTAL_HEALTH_EMOTIONS = ['anger', 'joy', 'optimism', 'sadness', 'fear', 'neutral']

class EmotionDetector:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.emotions = MENTAL_HEALTH_EMOTIONS  # Use our mental health emotions
        self._load_model()

    def _load_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import os

            # Try to load fine-tuned model first
            if os.path.exists(FINE_TUNED_MODEL_PATH):
                print(f"Loading fine-tuned mental health model from {FINE_TUNED_MODEL_PATH}")
                self.tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_PATH)
                self.model = AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_MODEL_PATH)
                print("✅ Fine-tuned model loaded successfully")
            else:
                print(f"Fine-tuned model not found at {FINE_TUNED_MODEL_PATH}")
                print(f"Loading fallback model: {FALLBACK_MODEL_NAME}")
                self.tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_NAME)
                self.model = AutoModelForSequenceClassification.from_pretrained(FALLBACK_MODEL_NAME)
                print("✅ Fallback model loaded successfully")

        except ImportError:
            print("Transformers not installed. Using fallback emotion detection.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to keyword-based detection.")

    def detect_emotion(self, text):
        if not self.model or not self.tokenizer:
            # Fallback: simple keyword-based emotion detection
            return self._fallback_emotion_detection(text)

        try:
            import torch
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            emotion = self.emotions[predicted_class]
            return emotion, confidence
        except Exception as e:
            print(f"Error detecting emotion: {e}")
            return self._fallback_emotion_detection(text)

    def _fallback_emotion_detection(self, text):
        text_lower = text.lower()
        if any(word in text_lower for word in ['happy', 'joy', 'great', 'excellent', 'good']):
            return 'joy', 0.8
        elif any(word in text_lower for word in ['sad', 'depressed', 'unhappy', 'terrible']):
            return 'sadness', 0.8
        elif any(word in text_lower for word in ['angry', 'mad', 'furious', 'hate']):
            return 'anger', 0.8
        elif any(word in text_lower for word in ['hope', 'optimistic', 'positive', 'better']):
            return 'optimism', 0.7
        else:
            return 'joy', 0.5  # default to positive for mental health

class ConversationalAI:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import os

            if os.path.exists(CONVERSATIONAL_MODEL_PATH):
                print(f"Loading conversational AI model from {CONVERSATIONAL_MODEL_PATH}")
                self.tokenizer = AutoTokenizer.from_pretrained(CONVERSATIONAL_MODEL_PATH)
                self.model = AutoModelForCausalLM.from_pretrained(CONVERSATIONAL_MODEL_PATH)
                print("✅ Conversational model loaded successfully")
            else:
                print(f"Conversational model not found at {CONVERSATIONAL_MODEL_PATH}")
                self.model = None

        except ImportError:
            print("Transformers not installed.")
            self.model = None
        except Exception as e:
            print(f"Error loading conversational model: {e}")
            self.model = None

    def generate_response(self, message, max_length=100):
        if not self.model or not self.tokenizer:
            return "I'm here to listen. How are you feeling today?"

        try:
            import torch

            # Prepare input
            input_text = f"Human: {message}\nAssistant:"
            inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=len(inputs['input_ids'][0]) + max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode response
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the assistant's response
            response = full_output.split("Assistant:")[-1].strip()

            return response if response else "I'm here to help. What's on your mind?"

        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I'm having trouble responding right now. Can you tell me more about how you're feeling?"

class FacialEmotionDetector:
    def __init__(self):
        self.detector = None
        self._load_detector()

    def _load_detector(self):
        try:
            from fer import FER
            import cv2
            self.detector = FER(mtcnn=True)  # Use MTCNN for better face detection
            print("✅ Facial emotion detector loaded successfully")
        except ImportError:
            print("FER library not installed. Facial emotion detection will be unavailable.")
        except Exception as e:
            print(f"Error loading facial emotion detector: {e}")

    def detect_emotions(self, image_file):
        """
        Detect emotions from image file
        Returns: dict with emotion probabilities and primary emotion
        """
        if not self.detector:
            return {"error": "Facial emotion detector not available"}

        try:
            import cv2
            import numpy as np
            from PIL import Image
            import io

            # Convert uploaded file to OpenCV format
            image = Image.open(image_file)
            image = np.array(image)

            # Convert RGB to BGR for OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Detect emotions
            result = self.detector.detect_emotions(image)

            if not result:
                return {"error": "No face detected in image"}

            # Get the first face (assuming single face)
            emotions = result[0]['emotions']

            # Find primary emotion
            primary_emotion = max(emotions, key=emotions.get)
            confidence = emotions[primary_emotion]

            return {
                "emotions": emotions,
                "primary_emotion": primary_emotion,
                "confidence": round(confidence, 3),
                "all_emotions": {k: round(v, 3) for k, v in emotions.items()}
            }

        except Exception as e:
            print(f"Error detecting facial emotions: {e}")
            return {"error": f"Failed to process image: {str(e)}"}

# Global instances
emotion_detector = EmotionDetector()
conversational_ai = ConversationalAI()
facial_emotion_detector = FacialEmotionDetector()

def analyze_message(message, facial_image=None):
    """
    Analyze user message for emotion and severity, optionally including facial emotion analysis
    Returns: emotion, severity_score, ai_response, facial_data
    """
    # Text emotion analysis
    text_emotion, text_confidence = emotion_detector.detect_emotion(message)

    # Facial emotion analysis (if image provided)
    facial_data = None
    if facial_image:
        facial_data = facial_emotion_detector.detect_emotions(facial_image)
        if "error" not in facial_data:
            print(f"Facial emotions detected: {facial_data}")

    # Combine text and facial emotions for multimodal prediction
    final_emotion, final_severity = combine_emotions(text_emotion, text_confidence, facial_data)

    # Severity mapping based on emotion (higher for negative emotions)
    severity_map = {
        'anger': 0.8,      # High severity - anger can indicate distress
        'sadness': 0.9,    # Very high - sadness/depression needs attention
        'joy': 0.2,        # Low severity - positive emotion
        'optimism': 0.3,   # Low-medium - hopeful but monitor
        'fear': 0.85,      # High severity - fear/anxiety
        'neutral': 0.4,    # Medium severity
    }

    severity_score = severity_map.get(final_emotion, 0.5)

    # Generate AI response based on emotion and severity
    response_text = generate_response_text(final_emotion, severity_score, facial_data)

    # Try conversational AI if available
    try:
        conv_response = conversational_ai.generate_response(message)
        if conv_response and len(conv_response) > 10:  # If meaningful response
            response_text = conv_response
    except Exception as e:
        print(f"Conversational AI failed: {e}")

    # Create structured JSON response
    ai_response = {
        "sentiment_score": round(text_confidence * 100, 1),
        "stress_level": int(severity_score * 100),
        "anxiety_level": int(severity_score * 80),  # Estimate anxiety from severity
        "emotional_state": final_emotion.title(),
        "risk_level": "High" if severity_score > 0.8 else "Moderate" if severity_score > 0.6 else "Low",
        "explanation": f"Detected {final_emotion} with {round(text_confidence * 100, 1)}% confidence",
        "multimodal": facial_data is not None,
        "recommendations": [
            "Practice deep breathing exercises",
            "Consider journaling your thoughts",
            "Reach out to trusted friends or family",
            "Maintain a regular sleep schedule"
        ],
        "doctor_recommendation": "Yes" if severity_score > 0.7 else "No",
        "safety_message": "If you're in crisis, please contact emergency services immediately." if severity_score > 0.8 else None,
        "response_text": response_text
    }

    return final_emotion, severity_score, json.dumps(ai_response), facial_data

def combine_emotions(text_emotion, text_confidence, facial_data):
    """
    Combine text and facial emotion analysis for multimodal prediction
    """
    if not facial_data or "error" in facial_data:
        return text_emotion, text_confidence

    # Simple fusion: if facial confidence > 0.6, give it more weight
    facial_emotion = facial_data.get('primary_emotion', text_emotion)
    facial_confidence = facial_data.get('confidence', 0)

    if facial_confidence > 0.6:
        # Use facial emotion if confidence is high
        return facial_emotion, facial_confidence
    else:
        # Stick with text emotion but adjust confidence slightly
        return text_emotion, min(text_confidence + 0.1, 1.0)

def generate_response_text(emotion, severity_score, facial_data=None):
    """
    Generate appropriate response text based on emotion and facial data
    """
    response_text = ""

    # Check if facial data indicates mismatch with text
    facial_emotion = facial_data.get('primary_emotion') if facial_data and "error" not in facial_data else None

    if facial_emotion and facial_emotion != emotion:
        response_text = f"I notice your words suggest {emotion}, but your facial expression shows {facial_emotion}. It's interesting how our internal feelings and external expressions can sometimes differ. "
    elif facial_data and "error" not in facial_data:
        response_text = f"Your facial expression confirms the {emotion} I detected in your message. "

    if emotion == 'sadness' and severity_score > 0.7:
        response_text += "I hear that you're feeling sad. It's brave of you to share this. Would you like me to connect you with a mental health professional who can provide more support?"
    elif emotion == 'anger' and severity_score > 0.7:
        response_text += "I sense some anger in your message. It's okay to feel angry sometimes. If this is affecting your daily life, talking to a professional might help."
    elif emotion == 'fear' and severity_score > 0.7:
        response_text += "I detect some fear or anxiety in your expression and words. Remember that you're safe here, and it's okay to feel this way. Professional support can help you navigate these feelings."
    elif emotion == 'joy':
        response_text += "I'm glad to hear you're feeling positive! Keep nurturing those good feelings. Is there anything specific that's bringing you joy today?"
    elif emotion == 'optimism':
        response_text += "That's great to hear some optimism! Hope is a powerful thing. How can I support you in maintaining this positive outlook?"
    else:
        response_text += f"I understand you're experiencing {emotion}. Your feelings are valid. How can I help you right now?"

    return response_text

def calculate_final_evaluation(conversation):
    """
    Calculate final evaluation for a conversation based on all chat logs.
    For simplicity, return the last message's evaluation.
    """
    last_chatlog = conversation.chatlog_set.last()
    if last_chatlog and last_chatlog.parsed_response:
        return json.loads(last_chatlog.parsed_response)
    return None

def recommend_doctor(emotion, location=None):
    """
    Recommend doctors based on detected emotion
    """
    emotion_specialization_map = {
        "sadness": "Clinical Psychologist",  # For depression-related sadness
        "anger": "Clinical Psychologist",    # For anger management
        "joy": "Life Coach",                 # For maintaining positive mental health
        "optimism": "Counselor",             # For general support
    }

    specialization = emotion_specialization_map.get(emotion, "Mental Health Professional")

    # In real implementation, this would query the database
    # For now, return a recommendation message
    return f"Based on your current emotional state, I recommend consulting a {specialization}. They can provide specialized support for {emotion}-related concerns."