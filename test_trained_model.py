import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_trained_model(model_path="./trained_sentiment_model"):
    """Load the trained model and tokenizer"""
    print("Loading trained model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer


def predict_sentiment(texts, model, tokenizer):
    """Predict sentiment for a list of texts"""
    # Tokenize the texts
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    
    # Get predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
    
    # Convert to labels
    results = []
    for text, pred in zip(texts, predictions):
        sentiment = "Positive" if pred.item() == 1 else "Negative"
        results.append((text, sentiment))
    
    return results


def main():
    """Main function to test the trained model"""
    try:
        # Load trained model
        model, tokenizer = load_trained_model()
        
        # Test with new sentences
        test_texts = [
            "This is a fantastic movie!",
            "I really disliked this film.",
            "It's an average movie, nothing special.",
            "Amazing acting and great storyline!",
            "Terrible movie, waste of time."
        ]
        
        print("\nTesting trained model with new sentences:")
        results = predict_sentiment(test_texts, model, tokenizer)
        
        print("\nSentiment Analysis Results:")
        print("-" * 50)
        for text, sentiment in results:
            print(f"{sentiment:10} | {text}")
            
    except Exception as e:
        print(f"Error loading or using trained model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()