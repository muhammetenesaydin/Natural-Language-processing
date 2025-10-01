import os
os.environ['TRANSFORMERS_NO_TF'] = '1'

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_model_and_tokenizer():
    """Load the DistilBERT model and tokenizer for sentiment analysis"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    
    return model, tokenizer


def predict_sentiment(texts, model, tokenizer):
    """Predict sentiment for a list of texts"""
    # Tokenize the texts
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(**encodings)
        predictions = torch.argmax(outputs.logits, dim=1)
    
    # Convert to labels
    results = []
    for text, pred in zip(texts, predictions):
        sentiment = "Positive" if pred.item() == 1 else "Negative"
        results.append((text, sentiment))
    
    return results


def main():
    """Main function to run sentiment analysis on sample texts"""
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
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