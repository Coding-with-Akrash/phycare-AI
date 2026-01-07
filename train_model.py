"""
Conversational AI Chatbot Training Script

This script fine-tunes a pre-trained conversational model (DialoGPT) on custom conversation data
to create a general-purpose chatbot that can respond to various user inputs.

For FYP demonstration, this shows how to train a conversational AI model.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from sklearn.model_selection import train_test_split
import numpy as np

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# Load conversational dataset from CSV file
def load_conversational_dataset(csv_path='data/conversational_data.csv'):
    """
    Load conversational dataset from CSV file
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with columns: {list(df.columns)}")
        print(f"DataFrame shape: {df.shape}")

        conversations = []
        for _, row in df.iterrows():
            # Handle different column names
            input_text = row.get('text') or row.get('input') or str(row.iloc[0])
            response_text = row.get('response') or str(row.iloc[1]) if len(row) > 1 else "I'm here to help."

            conversations.append({
                "input": str(input_text),
                "response": str(response_text)
            })

        print(f" Successfully loaded {len(conversations)} conversation pairs from {csv_path}")
        return conversations

    except FileNotFoundError:
        print(f" Dataset file not found: {csv_path}")
        print("Using built-in mental health conversation samples...")
        return SAMPLE_CONVERSATIONS
    except Exception as e:
        print(f" Error loading dataset: {e}")
        print("Using built-in mental health conversation samples...")
        return SAMPLE_CONVERSATIONS

# Sample conversations (fallback) - Mental Health Focused
SAMPLE_CONVERSATIONS = [
    {"input": "Hello, how are you?", "response": "Hi there! I'm doing well, thank you for asking. How are you feeling today?"},
    {"input": "What's your name?", "response": "I'm an AI assistant designed to help with mental health and wellness. You can call me PsyCare!"},
    {"input": "Tell me a joke", "response": "Why don't scientists trust atoms? Because they make up everything! 😄 I hope that brought a smile to your face."},
    {"input": "I'm feeling sad", "response": "I'm sorry to hear that you're feeling sad. It's completely normal to have these feelings. Would you like to talk about what's making you feel this way?"},
    {"input": "What is AI?", "response": "AI stands for Artificial Intelligence. It's technology that helps machines learn and make decisions, like me helping you with your mental health journey."},
    {"input": "I need help with coding", "response": "Sure! I'd be happy to help with coding. What programming language are you working with, and what specific problem are you facing?"},
    {"input": "Good morning", "response": "Good morning! I hope you have a positive and productive day ahead. Remember to take care of yourself."},
    {"input": "Good night", "response": "Good night! Sleep well and take care. Remember, tomorrow is a new day with new possibilities."},
    {"input": "I'm feeling anxious", "response": "Anxiety can be really challenging. Try taking some deep breaths - inhale for 4 counts, hold for 4, exhale for 4. Would you like to talk about what's causing your anxiety?"},
    {"input": "I feel stressed", "response": "Stress is something many people deal with. What aspects of your life are feeling most stressful right now? Sometimes talking about it can help."},
    {"input": "I'm depressed", "response": "I'm really sorry you're feeling this way. Depression can make everything feel overwhelming. Have you considered speaking with a mental health professional?"},
    {"input": "I can't sleep", "response": "Sleep difficulties can really affect mental health. Have you tried establishing a bedtime routine or creating a calm sleep environment?"},
    {"input": "I feel lonely", "response": "Feeling lonely is tough, and it's something many people experience. Would you like to talk about ways to connect with others or build social support?"},
    {"input": "I'm having panic attacks", "response": "Panic attacks can be frightening. Remember that they are temporary and will pass. Try grounding techniques - name 5 things you can see, 4 you can touch, etc."},
    {"input": "I need motivation", "response": "Finding motivation can be challenging. What are some small, achievable goals you could work toward today? Remember, progress is more important than perfection."},
    {"input": "I feel worthless", "response": "Those feelings of worthlessness can be very painful. Please know that you have inherent value as a person. Your feelings don't define your worth."},
    {"input": "I'm scared of the future", "response": "Fear of the future is common, especially during uncertain times. What specific worries do you have? Breaking them down can sometimes make them feel more manageable."},
    {"input": "I have low self-esteem", "response": "Low self-esteem can make everything feel harder. Try practicing self-compassion. What are some things you like about yourself or accomplishments you're proud of?"},
    {"input": "I feel overwhelmed", "response": "Feeling overwhelmed is a sign that you might need to prioritize or break things down. What tasks are feeling most overwhelming right now?"},
    {"input": "I need someone to talk to", "response": "I'm here to listen. What's on your mind? Sometimes just having someone to talk to can make a big difference."},
    {"input": "I'm having suicidal thoughts", "response": "I'm really concerned about what you're telling me. Please reach out to emergency services or a crisis hotline immediately. You don't have to face this alone."},
    {"input": "I feel angry", "response": "Anger is a valid emotion. What do you think is causing these feelings of anger? Understanding the root can help in managing it."},
    {"input": "I have social anxiety", "response": "Social anxiety can make social situations feel terrifying. Gradual exposure and relaxation techniques can help. Have you found any strategies that work for you?"},
    {"input": "I feel guilty", "response": "Feelings of guilt can be heavy to carry. What are you feeling guilty about? Sometimes examining these feelings can help process them."},
    {"input": "I need coping strategies", "response": "There are many coping strategies that can help. Some include deep breathing, journaling, exercise, mindfulness, or talking to trusted friends. What works best for you?"},
    {"input": "I'm grieving", "response": "Grieving is a natural process after loss. It's okay to feel all the emotions that come with it. Give yourself time and be gentle with yourself."},
    {"input": "I feel hopeless", "response": "Feeling hopeless can be a symptom of depression. While it might not feel like it now, things can and do get better. Have you considered professional help?"},
    {"input": "I have trust issues", "response": "Trust issues often develop from past experiences. Building trust takes time and can be challenging. What experiences have led to these feelings?"},
    {"input": "I'm having relationship problems", "response": "Relationship difficulties can be really stressful. Communication is often key. What specific issues are you facing in your relationships?"},
    {"input": "I feel empty inside", "response": "Feeling empty inside can be a sign of depression or emotional burnout. What activities used to bring you joy? Sometimes reconnecting with those can help."},
]

class ConversationalDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_length=512):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conversation = self.conversations[idx]

        # Format as a conversation
        text = f"User: {conversation['input']}\nAssistant: {conversation['response']}"

        # Ensure text is properly formatted
        if not isinstance(text, str) or len(text.strip()) == 0:
            text = "User: Hello\nAssistant: Hi there!"

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
            'labels': encoding['input_ids'].flatten()  # For causal LM, labels are the same as input_ids
        }

def train_conversational_model():
    """
    Fine-tune RoBERTa-base model on conversational data
    """
    print(" Starting Conversational AI Model Training")
    print("=" * 60)

    # Load pre-trained conversational model
    model_name = 'microsoft/DialoGPT-small'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Load dataset
    conversations = load_conversational_dataset()
    df = pd.DataFrame(conversations)

    # Split dataset
    train_conversations, val_conversations = train_test_split(
        conversations,
        test_size=0.2,
        random_state=42
    )

    # Create datasets
    train_dataset = ConversationalDataset(train_conversations, tokenizer)
    val_dataset = ConversationalDataset(val_conversations, tokenizer)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Not using masked language modeling
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./conversational_ai_model',
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Smaller batch size for CPU/memory constraints
        per_device_eval_batch_size=2,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # Lower loss is better
        save_total_limit=2,  # Keep only the best 2 checkpoints
        no_cuda=not torch.cuda.is_available(),  # Disable CUDA if not available
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("\n Starting training...")
    trainer.train()

    print("\n Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation Results: {eval_results}")

    # Save the fine-tuned model
    print("\n Saving fine-tuned conversational model...")
    trainer.save_model('./conversational_ai_model')
    tokenizer.save_pretrained('./conversational_ai_model')

    print(" Model training completed!")
    print(" Model saved to: ./conversational_ai_model")
    print("\nTo use this model in production, update the model path in ai_utils.py")

    return trainer

def test_conversational_model():
    """
    Test the trained conversational model with sample inputs
    """
    print("\n Testing conversational model...")

    # Load the trained model
    model_path = './conversational_ai_model'
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

        # Add padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        test_inputs = [
            "Hello, how are you?",
            "I'm feeling sad today",
            "Tell me a joke",
            "What is AI?",
            "I need help with coding"
        ]

        for user_input in test_inputs:
            # Format input
            input_text = f"User: {user_input}\nAssistant:"

            # Tokenize
            input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

            # Generate response
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_length=100,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.8,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Decode response
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

            # Extract just the assistant's response
            if "Assistant:" in generated_text:
                response = generated_text.split("Assistant:")[-1].strip()
            else:
                response = generated_text.replace(input_text, "").strip()

            print(f"User: {user_input}")
            print(f"Assistant: {response}")
            print("-" * 80)

    except Exception as e:
        print(f"Error testing model: {e}")
        print("Make sure the model has been trained first.")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_conversational_model()
    else:
        train_conversational_model()
        test_conversational_model()