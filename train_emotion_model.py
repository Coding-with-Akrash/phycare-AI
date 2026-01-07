"""
Mental Health Emotion Detection Model Training Script

This script fine-tunes a RoBERTa model on mental health emotion data
to create a specialized emotion detection model for the PsyCare AI platform.

The model is trained to classify text into 6 emotion categories:
- joy
- sadness
- anger
- optimism
- fear
- neutral
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# Emotion labels (must match ai_utils.py)
EMOTION_LABELS = ['anger', 'joy', 'optimism', 'sadness', 'fear', 'neutral']
label2id = {label: i for i, label in enumerate(EMOTION_LABELS)}
id2label = {i: label for i, label in enumerate(EMOTION_LABELS)}

def load_emotion_data(csv_path='data/mental_health_emotions_combined.csv'):
    """
    Load emotion dataset from CSV file
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with columns: {list(df.columns)}")
        print(f"DataFrame shape: {df.shape}")

        # Filter to only include our target emotions
        df = df[df['emotion'].isin(EMOTION_LABELS)]
        print(f"After filtering to target emotions: {df.shape}")

        # Convert emotion labels to IDs
        df['label'] = df['emotion'].map(label2id)

        # Check class distribution
        print("\nClass distribution:")
        print(df['emotion'].value_counts())

        return df

    except FileNotFoundError:
        print(f"Dataset file not found: {csv_path}")
        raise
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_emotion_model():
    """
    Fine-tune RoBERTa model for emotion detection
    """
    print("🚀 Starting Mental Health Emotion Detection Model Training")
    print("=" * 70)

    # Load dataset
    df = load_emotion_data()

    # Split dataset
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['emotion']  # Ensure balanced split
    )

    print(f"\nTraining samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    # Load pre-trained model and tokenizer
    model_name = 'roberta-base'
    print(f"\nLoading pre-trained model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(EMOTION_LABELS),
        id2label=id2label,
        label2id=label2id
    ).to(device)

    # Create datasets
    train_dataset = EmotionDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        tokenizer
    )

    val_dataset = EmotionDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        tokenizer
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./mental_health_emotion_model',
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        learning_rate=2e-5,
        no_cuda=not torch.cuda.is_available(),
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\n🏃 Starting training...")
    trainer.train()

    print("\n📊 Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation Results: {eval_results}")

    # Save the fine-tuned model
    print("\n💾 Saving fine-tuned emotion detection model...")
    trainer.save_model('./mental_health_emotion_model')
    tokenizer.save_pretrained('./mental_health_emotion_model')

    # Save training arguments for reference
    with open('./mental_health_emotion_model/training_info.txt', 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: mental_health_emotions_combined.csv\n")
        f.write(f"Training samples: {len(train_df)}\n")
        f.write(f"Validation samples: {len(val_df)}\n")
        f.write(f"Emotions: {', '.join(EMOTION_LABELS)}\n")
        f.write(f"Best F1 Score: {eval_results.get('eval_f1', 'N/A')}\n")
        f.write(f"Accuracy: {eval_results.get('eval_accuracy', 'N/A')}\n")

    print("✅ Model training completed!")
    print("📁 Model saved to: ./mental_health_emotion_model")
    print("\n🎯 The model is now ready for use in ai_utils.py")

    return trainer

def test_emotion_model():
    """
    Test the trained emotion model with sample inputs
    """
    print("\n🧪 Testing emotion detection model...")

    model_path = './mental_health_emotion_model'
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

        test_texts = [
            "I feel so happy and excited today!",
            "I'm really sad and don't know what to do",
            "I'm so angry at everything right now",
            "I feel hopeful about the future",
            "I'm scared of what might happen next",
            "Today is just an ordinary day",
            "I'm furious about this situation",
            "I believe things will get better soon"
        ]

        print("\nTest Results:")
        print("-" * 80)

        for text in test_texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            emotion = EMOTION_LABELS[predicted_class]
            print(f"Text: {text}")
            print(f"Predicted Emotion: {emotion} (Confidence: {confidence:.2%})")
            print("-" * 80)

    except Exception as e:
        print(f"❌ Error testing model: {e}")
        print("Make sure the model has been trained first.")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_emotion_model()
    else:
        train_emotion_model()
        test_emotion_model()