import os
os.environ['TRANSFORMERS_NO_TF'] = '1'

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset


def create_sample_data():
    """Create sample data for demonstration"""
    # Sample texts and labels for training
    train_texts = [
        "I love this movie, it's fantastic!",
        "This film is terrible, I hate it.",
        "Great acting and wonderful story.",
        "Worst movie ever, complete waste of time.",
        "Amazing cinematography and direction.",
        "Boring plot and bad acting.",
        "Excellent movie, highly recommended!",
        "Awful film, don't watch it.",
        "Brilliant performance by the actors.",
        "Poor script and terrible execution."
    ]
    
    train_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative
    
    # Validation data
    val_texts = [
        "This movie is okay, not great but not bad.",
        "I enjoyed watching this film.",
        "Not my cup of tea, quite disappointing.",
        "Decent movie with good moments."
    ]
    
    val_labels = [1, 1, 0, 1]
    
    return train_texts, train_labels, val_texts, val_labels


def tokenize_data(train_texts, train_labels, val_texts, val_labels):
    """Tokenize the text data using DistilBERT tokenizer"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    print("Tokenizing data...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
    
    return train_encodings, val_encodings, tokenizer


class SentimentDataset(torch.utils.data.Dataset):
    """Custom Dataset class for sentiment data"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


def load_model():
    """Load the DistilBERT model for sequence classification"""
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    return model


def train_model_manual(model, train_loader, val_loader, epochs=3):
    """Train the model manually without using Trainer"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    model.train()
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if num_batches % 5 == 0:
                print(f"  Batch {num_batches}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"  Average training loss: {avg_loss:.4f}")
        
        # Validation
        evaluate_model(model, val_loader, device)
    
    return model


def evaluate_model(model, val_loader, device):
    """Evaluate the model on validation set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total if total > 0 else 0
    print(f"  Validation Accuracy: {accuracy:.4f} ({correct}/{total})")
    model.train()


def save_model(model, tokenizer, save_path="./trained_sentiment_model"):
    """Save the trained model and tokenizer"""
    print(f"\nSaving model to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Model saved successfully!")


def test_model(model, tokenizer, texts):
    """Test the model with sample texts"""
    # Tokenize the texts
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    
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
    """Main function to run the sentiment analysis training"""
    try:
        print("=== Sentiment Analysis Training ===")
        
        # Create sample data
        train_texts, train_labels, val_texts, val_labels = create_sample_data()
        print(f"Train samples: {len(train_texts)}, Validation samples: {len(val_texts)}")
        
        # Tokenize data
        train_encodings, val_encodings, tokenizer = tokenize_data(
            train_texts, train_labels, val_texts, val_labels
        )
        
        # Create datasets
        train_dataset = SentimentDataset(train_encodings, train_labels)
        val_dataset = SentimentDataset(val_encodings, val_labels)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)
        
        # Load model
        model = load_model()
        
        # Train model
        print("\nStarting training...")
        trained_model = train_model_manual(model, train_loader, val_loader, epochs=5)
        
        # Save trained model
        save_model(trained_model, tokenizer)
        
        # Test with sample sentences
        test_texts = [
            "This movie is absolutely wonderful!",
            "I didn't like this film at all.",
            "It's an okay movie, nothing special."
        ]
        
        print("\nTesting with sample sentences:")
        results = test_model(trained_model, tokenizer, test_texts)
        
        print("\nSentiment Analysis Results:")
        print("-" * 50)
        for text, sentiment in results:
            print(f"{sentiment:10} | {text}")
        
        print("\nTraining completed successfully!")
        print("Model saved to './trained_sentiment_model'")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()