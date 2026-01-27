# PyTorch-based Sentiment Analysis
# This script uses a PyTorch-based model for sentiment analysis to avoid TensorFlow issues

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

def main():
    print("PyTorch-based Sentiment Analysis")
    print("=" * 30)
    
    # Use a PyTorch-based sentiment analysis pipeline
    # Specify the framework to avoid TensorFlow
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
            framework="pt"  # Explicitly use PyTorch
        )
        
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
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Falling back to manual approach...")
        
        # Fallback approach using manual model loading
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Sample texts
        texts = [
            "I loved this movie! It was amazing.",
            "This was a terrible film. I hated it."
        ]
        
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class_id = torch.argmax(predictions, dim=1).item()
                
                # The model uses different labels
                labels = ["Negative", "Neutral", "Positive"]
                label = labels[predicted_class_id]
                confidence = predictions[0][predicted_class_id].item() * 100
                
                print(f"\nText: {text}")
                print(f"Sentiment: {label} (Confidence: {confidence:.2f}%)")

if __name__ == "__main__":
    main()