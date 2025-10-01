"""
Complete Sentiment Analysis Workflow
Tamamlanmış duygu analizi iş akışı 
===================================

This script demonstrates the complete workflow for training a sentiment analysis model
and using it for inference.
Bu betik duygu analizi modelinin eğitilmesi ve çıkarım yapması için  yazılmış bir iş akışıdır.

1. Training phase: Train a model on sample data
2. Saving phase: Save the trained model
3. Inference phase: Load the trained model and use it for predictions

1- Eğitim bölümü: bu bölümde model eğitimi yapılıyor.
2- Kayıt Bölümü: bu bölümde model kayediliyor.
3- Çıkarım Bölümü: bu bölümde eğitilmiş model yeniden yükleip tahmin için kullanılıyor.
"""

import os
#tensorflow backendi tamamen kısıtlandı. (benim bilgisayarımda hem tensorflow hem torch kurulu olduğu için çakışma oluyordu çözümü bu şekilde.)
os.environ['TRANSFORMERS_NO_TF'] = '1'

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# === TRAINING PHASE ==EĞİTİM BÖLÜMÜ===

class SentimentDataset(torch.utils.data.Dataset):
    """
    A custom Dataset class for PyTorch.
    It combines the encodings (input_ids, attention_mask, etc.) 
    from the Tokenizer with the labels and returns them in a format that the DataLoader can use.

    
    PyTorch için özel Dataset sınıfı.
    Tokenizer'dan gelen encodings (input_ids, attention_mask vb.)
    ile etiketleri (labels) birleştirip DataLoader'ın kullanabileceği
    formatta döndürür.
    """
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


def create_training_data():
    """Create training and validation data
    
       Eğitim ve doğrulama verilerinin üretilmesi.   
    """
    # Training data , Eğitim verisi
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
        "Poor script and terrible execution.",
        "Outstanding film with great characters.",
        "Disappointing movie with weak plot.",
        "Incredible storytelling and visuals.",
        "Dull and uninteresting film.",
        "Masterpiece of cinema!",
        "Complete garbage, avoid at all costs."
    ]
    # positives are labeled 1 and negatives are labeled 0 
    # pozitifler 1 negatifler 0 şeklinde etiketlenmiş 
    train_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    
    # Validation data Doğrulama verisi 
    val_texts = [
        "This movie is okay, not great but not bad.",
        "I enjoyed watching this film.",
        "Not my cup of tea, quite disappointing.",
        "Decent movie with good moments."
    ]
    
    val_labels = [1, 1, 0, 1]
    
    return train_texts, train_labels, val_texts, val_labels


def tokenize_data(train_texts, train_labels, val_texts, val_labels):
    """Tokenize the text data Veriyi tokenize eder."""

    # Here it loads the tokenizer of the distilbert-base-uncased model.
    # Burada distilbert-base-uncased modelinin tokenizerini yüklüyor. 
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Burada eğitim ve doğrulama verileri tokenizerden geçirilir.
    # Here the train and validation data is passed through the tokenizer 
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
    
    return train_encodings, val_encodings, tokenizer


def train_model():
    """Train a sentiment analysis model
       Duygu analizi modeli eğitilir.
    """
    
    print("=== TRAINING PHASE ===")

    # Data is retrieved from create_data()
    # create_data() dan veriler çekilir.
    train_texts, train_labels, val_texts, val_labels = create_training_data()
    print(f"Training samples: {len(train_texts)}, Validation samples: {len(val_texts)}")
    
    # Tokenize data
    # Veriyi tokenize eder.
    # train_encodings /val_encodings: metinlerin token ID’leri + attention mask’leri
    train_encodings, val_encodings, tokenizer = tokenize_data(
        train_texts, train_labels, val_texts, val_labels
    )
    
    # Create datasets
    # encodings + labels/etiketler -->pytorch dataset 
    train_dataset = SentimentDataset(train_encodings, train_labels)
    val_dataset = SentimentDataset(val_encodings, val_labels)
    
    # Create data loaders 
    # batch_size=4, 4 örnek bir seferde işleniyor shuffle trainde true  burada veriler rastgele alınacak 
    # doğrulamada false doğrulama verisi sırayla alınacak. 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Load model
    # Model Yüklenir. dizi sınıflandırma modeli kullanılıyor. label 2 pozitif veya negatif.
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    
    # Training setup
    # Eğitim Ortamı
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)

    # optimizasyon algoritması kullanılıyor. AdamW  en yaygın olanlardan.
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Training loop
    # eğitim döngüsü
    model.train()
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}/3")
        total_loss = 0 # toplam kayıp 
        num_batches = 0 # toplam kaç batch işlendiğini görek çin kullanılır.
        
        for batch in train_loader:
            # her batch içindeki inpu_ids atention_mask labels (işlem birimine taşınır (cpu/gpu))
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            # model girişleri işler girişleri işler çıktıyı döner. 
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1
            
            # Backward pass
            
            optimizer.zero_grad() # gradyanları sıfırlar
            loss.backward() # gradyan hesaplar 
            optimizer.step() # modelin ağırlıklarını günceller.

        # epoch bittiğinde ortalam loss hesaplanıyor ve yazdırılıyor 
        avg_loss = total_loss / num_batches
        print(f"  Average training loss: {avg_loss:.4f}")
    
    return model, tokenizer


def save_model(model, tokenizer, save_path="./sentiment_model"):
    """Save the trained model and tokenizer
       Modeli  ve tokenizeri kaydet.   
    """
    print(f"\n=== SAVING PHASE ===")
    print(f"Saving model to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Model saved successfully!")


# === INFERENCE PHASE ===Çıkarım Bölümü===

def load_model(model_path="./sentiment_model"):
    """Load the trained model and tokenizer
       Modeli ve tokenizeri içeri aktarır   
    """
    print(f"\n=== INFERENCE PHASE ===")
    print("Loading trained model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer


def predict_sentiment(texts, model, tokenizer):
    """Predict sentiment for a list of texts
       Listedeki cümlelerin sentimentlerini tahmin edelim.  
    """
    # Tokenize the texts
    # cümleleri Tokenize edelim
    # çıktıyı pytorch tensorleri olarak döndürür.
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    
    # Get predictions
    # Tahminleri yapalım
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval() # evaluation mode dropout off

    
    with torch.no_grad(): #inference sırasında gradient hesaplamayı kapatır.(hız+bellek tasarrufu) 
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)  # Output logits modelin her bir sınıf için verdiği skorlardır.
        #dim=1  en yüksek skora sahip sınıf seçilir (pozitif negatif) şeklinde.
    
    # Convert to labels
    
    results = []
    for text, pred in zip(texts, predictions):
        sentiment = "Positive" if pred.item() == 1 else "Negative" # modelin sınıf tahmini burada yapılır.etiket burada atanır. 
        confidence = torch.softmax(outputs.logits, dim=1)[0][pred].item() #tahmin edilen sınıfın olasılığı
        results.append((text, sentiment, confidence)) # sonuç (metin, duygu, güven skoru.) 
    
    return results


def main():
    """Main function demonstrating the complete workflow"""
    print("Complete Sentiment Analysis Workflow")
    print("=" * 40)
    
    # 1. Training Phase 1. Bölüm 
    model, tokenizer = train_model()
    
    # 2. Saving Phase 2. Bölüm 
    save_model(model, tokenizer)
    
    # 3. Inference Phase  3. Bölüm 
    # Load the trained model
    model, tokenizer = load_model()
    
    # Test with sample sentences
    # Test için örnekler.
    test_texts = [
        "This movie is absolutely wonderful!",
        "I didn't like this film at all.",
        "It's an okay movie, nothing special.",
        "Amazing acting and great storyline!",
        "Terrible movie, complete waste of time."
    ]
    
    print("\nTesting with sample sentences:")
    results = predict_sentiment(test_texts, model, tokenizer)
    
    print("\nSentiment Analysis Results:")
    print("-" * 60)
    print(f"{'Sentiment':<12} {'Confidence':<12} {'Text'}")
    print("-" * 60)
    for text, sentiment, confidence in results:
        print(f"{sentiment:<12} {confidence:<12.4f} {text}")
    
    print("\nWorkflow completed successfully!")
    print("Model saved to './sentiment_model'")


if __name__ == "__main__":
    main()
