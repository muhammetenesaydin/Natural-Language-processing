import os
os.environ['TRANSFORMERS_NO_TF'] = '1'

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_pretrained_model():
    """Load a pre-trained sentiment analysis model"""
    print("Loading pre-trained sentiment analysis model...")
    # Using a model that's already fine-tuned for sentiment analysis
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer


def predict_sentiment(texts, model, tokenizer):
    """Predict sentiment for a list of texts using pre-trained model"""
    # Tokenize the texts
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(**encodings)
        predictions = torch.argmax(outputs.logits, dim=1)
    
    # Convert to labels
    label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    results = []
    for text, pred in zip(texts, predictions):
        sentiment = label_mapping.get(int(pred.item()), "Unknown")
        results.append((text, sentiment))
    
    return results


def main():
    """Main function to run sentiment analysis on sample texts"""
    # Load pre-trained model and tokenizer
    model, tokenizer = load_pretrained_model()
    
    # Sample texts for sentiment analysis
    sample_texts = [
        "I loved this movie! It was fantastic.",
        "This was a terrible film. I hated it.",
        "The acting was superb and the plot was engaging.",
        "Worst movie I've ever seen, complete waste of time.",
        "It was okay, nothing special but not bad either."
    ]
    
    print("\nAnalyzing sentiments...")
    results = predict_sentiment(sample_texts, model, tokenizer)
    
    print("\nSentiment Analysis Results:")
    print("-" * 50)
    for text, sentiment in results:
        print(f"{sentiment:10} | {text}")


if __name__ == "__main__":
    main()