# Simple NLP Sentiment Analysis with Transformers
# This script performs sentiment analysis on sample texts using DistilBERT

# Install required packages (uncomment if needed)
# !pip install transformers torch

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def analyze_sentiment(texts):
    """Analyze sentiment of given texts using DistilBERT"""
    # Load pre-trained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    
    # Tokenize texts
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**encodings)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_labels = torch.argmax(predictions, dim=1)
    
    return predicted_labels, predictions

def main():
    print("Simple Sentiment Analysis with DistilBERT")
    print("=" * 40)
    
    # Sample texts for sentiment analysis
    sample_texts = [
        "I loved this movie! It was amazing.",
        "This was a terrible film. I hated it.",
        "The movie was okay, nothing special.",
        "What a fantastic experience! Highly recommend.",
        "Worst movie ever. Complete waste of time."
    ]
    
    print("Analyzing sentiments...")
    labels, probabilities = analyze_sentiment(sample_texts)
    
    # Display results
    for text, label, probs in zip(sample_texts, labels, probabilities):
        sentiment = "Positive" if label.item() == 1 else "Negative"
        confidence = probs[label].item() * 100
        print(f"\nText: {text}")
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f}%)")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()