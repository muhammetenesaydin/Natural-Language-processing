# Pre-trained Sentiment Analysis with Transformers
# This script uses a model that's already trained for sentiment analysis

import torch
from transformers import pipeline

def main():
    print("Pre-trained Sentiment Analysis")
    print("=" * 30)
    
    # Use a pre-trained sentiment analysis pipeline
    # This model is already trained for sentiment analysis
    sentiment_pipeline = pipeline("sentiment-analysis")
    
    # Sample texts for sentiment analysis
    sample_texts = [
        "I loved this movie! It was amazing.",
        "This was a terrible film. I hated it.",
        "The movie was okay, nothing special.",
        "What a fantastic experience! Highly recommend.",
        "Worst movie ever. Complete waste of time."
    ]
    
    print("Analyzing sentiments...")
    results = sentiment_pipeline(sample_texts)
    
    # Display results
    for text, result in zip(sample_texts, results):
        print(f"\nText: {text}")
        print(f"Sentiment: {result['label']} (Confidence: {result['score']*100:.2f}%)")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()