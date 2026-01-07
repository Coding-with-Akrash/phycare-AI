from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Path to save the model
model_path = './mental_health_emotion_model'

# Model to download (using the fallback model as a base)
model_name = 'cardiffnlp/twitter-roberta-base-emotion'

print(f"Downloading model {model_name} to {model_path}")

# Download and save
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create directory if not exists
os.makedirs(model_path, exist_ok=True)

# Save
tokenizer.save_pretrained(model_path)
model.save_pretrained(model_path)

print("Model downloaded and saved successfully.")