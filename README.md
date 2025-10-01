# Sentiment Analysis with Pre-trained Models

This project implements sentiment analysis using transformer models to classify text as positive, negative, or neutral. It includes four main scripts:

This project provides various Python scripts for performing text-based sentiment analysis. It demonstrates both using pre-trained models and training your own models.

## Requirements

- Python 3.7+
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-learn
- Accelerate (for training)

## Installation

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv nlp_env
   source nlp_env/bin/activate  # On Windows: nlp_env\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

If you encounter issues with TensorFlow dependencies, you can install the packages individually:
```bash
pip install torch transformers datasets scikit-learn accelerate
```

## Usage

### Pre-trained Model (Recommended)

Run the pre-trained sentiment analysis script:
```bash
python pretrained_sentiment.py
```

This script will:
1. Load a pre-trained sentiment analysis model
2. Analyze sample sentences for sentiment
3. Provide accurate sentiment predictions

### Complete Workflow (Recommended for Learning)

Run the complete workflow script:
```bash
python complete_workflow.py
```

This script demonstrates the complete end-to-end workflow:
1. Training a model on sample data
2. Saving the trained model
3. Loading the trained model
4. Performing inference on new data

### Simple Inference (Educational)

Run the lightweight sentiment analysis script:
```bash
python simple_sentiment.py
```

This script will:
1. Load the DistilBERT model and tokenizer
2. Analyze sample sentences for sentiment

Note: This uses an untrained model with random weights, so predictions will be random.

### Training a Model (Educational)

Run the simplified training script (recommended for learning):
```bash
python final_train.py
```

This script will:
1. Create sample training data
2. Tokenize the data using DistilBERT tokenizer
3. Load the pre-trained DistilBERT model
4. Train the model using a manual training loop
5. Evaluate the model during training
6. Save the trained model to disk

Note: This is a simplified example for learning purposes.

## Customization

To train the model on the IMDb dataset, uncomment the training section in the `main()` function:
```python
# Train model (commented out as it takes time)
trainer = train_model(model, train_dataset, test_dataset)
```

Note that training can take several hours depending on your hardware.

## Sample Output

The pre-trained sentiment script will output accurate sentiment predictions for sample sentences:
```
Positive   | I loved this movie! It was fantastic.
Negative   | This was a terrible film. I hated it.
Positive   | The acting was superb and the plot was engaging.
Negative   | Worst movie I've ever seen, complete waste of time.
Positive   | It was okay, nothing special but not bad either.
```

The simple sentiment script (using untrained model) will output random predictions:
```
Positive   | I loved this movie! It was fantastic.
Positive   | This was a terrible film. I hated it.
Positive   | The acting was superb and the plot was engaging.
Positive   | Worst movie I've ever seen, complete waste of time.
Positive   | It was okay, nothing special but not bad either.
```

---

# Önceden Eğitilmiş Modellerle Duygu Analizi

Bu proje, metinleri pozitif, negatif veya nötr olarak sınıflandırmak için dönüştürücü modelleri kullanarak duygu analizi uygular. Dört ana betik içerir:

Bu proje, metin tabanlı duygu analizi yapmak için çeşitli Python betikleri sağlar. Hem önceden eğitilmiş modellerin kullanılmasını hem de kendi modellerinizin eğitilmesini gösterir.

## Gereksinimler

- Python 3.7+
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-learn
- Accelerate (eğitim için)

## Kurulum

1. Sanal bir ortam oluşturun (isteğe bağlı ancak önerilir):
   ```bash
   python -m venv nlp_env
   source nlp_env/bin/activate  # On Windows: nlp_env\Scripts\activate
   ```

2. Gerekli paketleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

TensorFlow bağımlılıklarıyla ilgili sorunlarla karşılaşırsanız, paketleri tek tek yükleyebilirsiniz:
```bash
pip install torch transformers datasets scikit-learn accelerate
```

## Kullanım

### Önceden Eğitilmiş Model (Önerilen)

Önceden eğitilmiş duygu analizi betiğini çalıştırın:
```bash
python pretrained_sentiment.py
```

Bu betik şu işlemleri yapar:
1. Önceden eğitilmiş bir duygu analizi modeli yükler
2. Örnek cümleleri duygu açısından analiz eder
3. Doğru duygu tahminleri sağlar

### Tam İş Akışı (Öğrenme için Önerilir)

Tam iş akışı betiğini çalıştırın:
```bash
python complete_workflow.py
```

Bu betik eksiksiz uçtan uca iş akışını gösterir:
1. Örnek veriler üzerinde bir modeli eğitmek
2. Eğitilen modeli kaydetmek
3. Eğitilen modeli yüklemek
4. Yeni veriler üzerinde çıkarım yapmak

### Basit Çıkarım (Eğitimsel)

Hafif duygu analizi betiğini çalıştırın:
```bash
python simple_sentiment.py
```

Bu betik şu işlemleri yapar:
1. DistilBERT modelini ve belirteçleştiriciyi yükler
2. Örnek cümleleri duygu açısından analiz eder

Not: Bu, rastgele ağırlıklara sahip eğitilmemiş bir model kullanır, bu yüzden tahminler rastgele olur.

### Model Eğitimi (Eğitimsel)

Basitleştirilmiş eğitim betiğini çalıştırın (öğrenme için önerilir):
```bash
python final_train.py
```

Bu betik şu işlemleri yapar:
1. Örnek eğitim verileri oluşturur
2. Verileri DistilBERT belirteçleştiriciyi kullanarak belirteçleştirir
3. Önceden eğitilmiş DistilBERT modelini yükler
4. Manuel eğitim döngüsünü kullanarak modeli eğitir
5. Eğitim sırasında modeli değerlendirir
6. Eğitilen modeli diske kaydeder

Not: Bu, öğrenme amaçlı basitleştirilmiş bir örnektir.

## Özelleştirme

IMDb veri kümesinde modeli eğitmek için `main()` fonksiyonundaki eğitim bölümünün yorumunu kaldırın:
```python
# Modeli eğit (zaman aldığı için yorum halinde)
trainer = train_model(model, train_dataset, test_dataset)
```

Eğitimin donanımınıza bağlı olarak birkaç saat sürebileceğini unutmayın.

## Örnek Çıktı

Önceden eğitilmiş duygu betiği, örnek cümleler için doğru duygu tahminleri çıkarır:
```
Positive   | I loved this movie! It was fantastic.
Negative   | This was a terrible film. I hated it.
Positive   | The acting was superb and the plot was engaging.
Negative   | Worst movie I've ever seen, complete waste of time.
Positive   | It was okay, nothing special but not bad either.
```

Basit duygu betiği (eğitilmemiş model kullanarak) rastgele tahminler çıkarır:
```
Positive   | I loved this movie! It was fantastic.
Positive   | This was a terrible film. I hated it.
Positive   | The acting was superb and the plot was engaging.
Positive   | Worst movie I've ever seen, complete waste of time.
Positive   | It was okay, nothing special but not bad either.
```
