# Simple PyTorch-based Sentiment Analysis
# This script uses a PyTorch-based model for sentiment analysis

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def analyze_sentiment(texts):
    """Analyze sentiment of texts using a pre-trained model"""
    # Load pre-trained model and tokenizer
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    results = []
    for text in texts:
        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class_id = int(torch.argmax(predictions, dim=1).item())
            confidence = float(predictions[0][predicted_class_id].item())
            
            # Convert to label
            labels = ["Negative", "Neutral", "Positive"]
            label = labels[predicted_class_id]
            
            results.append({
                "text": text,
                "label": label,
                "confidence": confidence
            })
    
    return results

def main():
    print("Simple PyTorch-based Sentiment Analysis")
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
    try:
        results = analyze_sentiment(sample_texts)
        
        # Display results
        for result in results:
            print(f"\nText: {result['text']}")
            print(f"Sentiment: {result['label']} (Confidence: {result['confidence']*100:.2f}%)")
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check your internet connection and try again.")

if __name__ == "__main__":
    main()