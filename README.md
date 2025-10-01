# Sentiment Analysis with Pre-trained Models
# Önceden Eğitilmiş Modellerle Duygu Analizi

This project implements sentiment analysis using transformer models to classify text as positive, negative, or neutral. It includes four main scripts:
Bu proje, metinleri pozitif, negatif veya nötr olarak sınıflandırmak için dönüştürücü modelleri kullanarak duygu analizi uygular. Dört ana betik içerir:

Bu proje, metin tabanlı duygu analizi yapmak için çeşitli Python betikleri sağlar. Hem önceden eğitilmiş modellerin kullanılmasını hem de kendi modellerinizin eğitilmesini gösterir.

This project provides various Python scripts for performing text-based sentiment analysis. It demonstrates both using pre-trained models and training your own models.

1. `pretrained_sentiment.py` - Uses a pre-trained sentiment analysis model (recommended for immediate use)
   `pretrained_sentiment.py` - Önceden eğitilmiş bir duygu analizi modeli kullanır (anında kullanım için önerilir)
2. `simple_sentiment.py` - Uses DistilBERT with random weights (for demonstration)
   `simple_sentiment.py` - Rastgele ağırlıklarla DistilBERT kullanır (demo için)
3. `final_train.py` - A simplified training script with manual training loop (recommended for learning)
   `final_train.py` - Manuel eğitim döngüsüyle basitleştirilmiş bir eğitim betiği (öğrenme için önerilir)
4. `complete_workflow.py` - A complete end-to-end workflow demonstrating training, saving, and inference
   `complete_workflow.py` - Eğitimi, kaydetmeyi ve çıkarımı gösteren eksiksiz uçtan uca bir iş akışı

## Requirements
## Gereksinimler

- Python 3.7+
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-learn
- Accelerate (for training)
- Accelerate (eğitim için)

## Installation
## Kurulum

1. Create a virtual environment (optional but recommended):
1. Sanal bir ortam oluşturun (isteğe bağlı ancak önerilir):
   ```bash
   python -m venv nlp_env
   source nlp_env/bin/activate  # On Windows: nlp_env\Scripts\activate
   ```

2. Install the required packages:
2. Gerekli paketleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

If you encounter issues with TensorFlow dependencies, you can install the packages individually:
TensorFlow bağımlılıklarıyla ilgili sorunlarla karşılaşırsanız, paketleri tek tek yükleyebilirsiniz:
```bash
pip install torch transformers datasets scikit-learn accelerate
```

## Usage
## Kullanım

### Pre-trained Model (Recommended)
### Önceden Eğitilmiş Model (Önerilen)

Run the pre-trained sentiment analysis script:
Önceden eğitilmiş duygu analizi betiğini çalıştırın:
```bash
python pretrained_sentiment.py
```

This script will:
Bu betik şu işlemleri yapar:
1. Load a pre-trained sentiment analysis model
   Önceden eğitilmiş bir duygu analizi modeli yükler
2. Analyze sample sentences for sentiment
   Örnek cümleleri duygu açısından analiz eder
3. Provide accurate sentiment predictions
   Doğru duygu tahminleri sağlar

### Complete Workflow (Recommended for Learning)
### Tam İş Akışı (Öğrenme için Önerilir)

Run the complete workflow script:
Tam iş akışı betiğini çalıştırın:
```bash
python complete_workflow.py
```

This script demonstrates the complete end-to-end workflow:
Bu betik eksiksiz uçtan uca iş akışını gösterir:
1. Training a model on sample data
   Örnek veriler üzerinde bir modeli eğitmek
2. Saving the trained model
   Eğitilen modeli kaydetmek
3. Loading the trained model
   Eğitilen modeli yüklemek
4. Performing inference on new data
   Yeni veriler üzerinde çıkarım yapmak

### Simple Inference (Educational)
### Basit Çıkarım (Eğitimsel)

Run the lightweight sentiment analysis script:
Hafif duygu analizi betiğini çalıştırın:
```bash
python simple_sentiment.py
```

This script will:
Bu betik şu işlemleri yapar:
1. Load the DistilBERT model and tokenizer
   DistilBERT modelini ve belirteçleştiriciyi yükler
2. Analyze sample sentences for sentiment
   Örnek cümleleri duygu açısından analiz eder

Note: This uses an untrained model with random weights, so predictions will be random.
Not: Bu, rastgele ağırlıklara sahip eğitilmemiş bir model kullanır, bu yüzden tahminler rastgele olur.

### Training a Model (Educational)
### Model Eğitimi (Eğitimsel)

Run the simplified training script (recommended for learning):
Basitleştirilmiş eğitim betiğini çalıştırın (öğrenme için önerilir):
```bash
python final_train.py
```

This script will:
Bu betik şu işlemleri yapar:
1. Create sample training data
   Örnek eğitim verileri oluşturur
2. Tokenize the data using DistilBERT tokenizer
   Verileri DistilBERT belirteçleştiriciyi kullanarak belirteçleştirir
3. Load the pre-trained DistilBERT model
   Önceden eğitilmiş DistilBERT modelini yükler
4. Train the model using a manual training loop
   Manuel eğitim döngüsünü kullanarak modeli eğitir
5. Evaluate the model during training
   Eğitim sırasında modeli değerlendirir
6. Save the trained model to disk
   Eğitilen modeli diske kaydeder

Note: This is a simplified example for learning purposes.
Not: Bu, öğrenme amaçlı basitleştirilmiş bir örnektir.

## Customization
## Özelleştirme

To train the model on the IMDb dataset, uncomment the training section in the `main()` function:
IMDb veri kümesinde modeli eğitmek için `main()` fonksiyonundaki eğitim bölümünün yorumunu kaldırın:
```python
# Train model (commented out as it takes time)
# Modeli eğit (zaman aldığı için yorum halinde)
trainer = train_model(model, train_dataset, test_dataset)
```

Note that training can take several hours depending on your hardware.
Eğitimin donanımınıza bağlı olarak birkaç saat sürebileceğini unutmayın.

## Sample Output
## Örnek Çıktı

The pre-trained sentiment script will output accurate sentiment predictions for sample sentences:
Önceden eğitilmiş duygu betiği, örnek cümleler için doğru duygu tahminleri çıkarır:
```
Positive   | I loved this movie! It was fantastic.
Negative   | This was a terrible film. I hated it.
Positive   | The acting was superb and the plot was engaging.
Negative   | Worst movie I've ever seen, complete waste of time.
Positive   | It was okay, nothing special but not bad either.
```

The simple sentiment script (using untrained model) will output random predictions:
Basit duygu betiği (eğitilmemiş model kullanarak) rastgele tahminler çıkarır:
```
Positive   | I loved this movie! It was fantastic.
Positive   | This was a terrible film. I hated it.
Positive   | The acting was superb and the plot was engaging.
Positive   | Worst movie I've ever seen, complete waste of time.
Positive   | It was okay, nothing special but not bad either.
```
