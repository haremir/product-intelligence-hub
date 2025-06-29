# 🛒 E-COMMERCE PRODUCT INTELLIGENCE HUB

## 📋 Proje Özeti / Project Overview

Bu proje, e-ticaret ürün verilerini kullanarak **ürün açıklaması üretimi** ve **kategori tahmini** yapan kapsamlı bir makine öğrenmesi çözümüdür. Proje, hem modüler yapıda hem de tek notebook formatında sunulmuştur.

This project is a comprehensive machine learning solution for **product description generation** and **category prediction** using e-commerce product data. The project is presented in both modular structure and single notebook format.

### 🔄 Neden İki Farklı Yaklaşım? / Why Two Different Approaches?

Bu projede hem **modüler yaklaşım** hem de **tek notebook yaklaşımı** kullanılmasının birkaç önemli nedeni vardır:

#### 📁 Modüler Yaklaşım (Modular Approach)
- **Geliştirici Deneyimi**: Kod organizasyonu ve bakım kolaylığı
- **Yeniden Kullanılabilirlik**: Modüllerin başka projelerde kullanılabilmesi
- **Test Edilebilirlik**: Her modülün ayrı ayrı test edilebilmesi
- **Takım Çalışması**: Farklı geliştiricilerin farklı modüller üzerinde çalışabilmesi
- **Ölçeklenebilirlik**: Büyük projeler için uygun yapı

#### 📓 Tek Notebook Yaklaşımı (Single Notebook Approach)
- **Hızlı Prototipleme**: Tek dosyada tüm işlemlerin görülebilmesi
- **Eğitim Amaçlı**: Öğrenme ve sunum için ideal format
- **Kolay Paylaşım**: Tek dosya olarak paylaşım kolaylığı
- **Akış Takibi**: Veri akışının adım adım takip edilebilmesi
- **Demo Amaçlı**: Hızlı demo ve gösterim için uygun

---

## 🎯 Ana Hedefler / Main Objectives

### 🤖 Generative AI (GenAI)
- Ürün başlığından otomatik açıklama üretimi
- HuggingFace T5-small modeli kullanımı
- Template-based fallback sistemi

### 🎯 Machine Learning (ML)
- Kategori tahmini için süpervizyonlu öğrenme
- TF-IDF vektörleştirme ile metin analizi
- Çoklu model karşılaştırması

### 📊 Data Analysis
- Kapsamlı keşifsel veri analizi (EDA)
- Veri görselleştirme ve istatistiksel analiz
- Veri kalitesi değerlendirmesi



## 🚀 Kurulum / Installation

### Gereksinimler / Requirements
- Python 3.13+
- uv (Python package manager)

### Kurulum Adımları / Installation Steps

```bash
# 1. Repository'yi klonlayın / Clone the repository
git clone <repository-url>
cd product-intelligence-hub

# 2. Sanal ortam oluşturun / Create virtual environment
uv venv

# 3. Bağımlılıkları yükleyin / Install dependencies
uv pip install -e .

# 4. Jupyter kernel'ı yükleyin / Install Jupyter kernel
uv pip install ipykernel
python -m ipykernel install --user --name=product-intelligence-hub
```

### Gerekli Kütüphaneler / Required Libraries

```toml
# pyproject.toml
[project]
dependencies = [
    "ipykernel>=6.29.5",
    "joblib>=1.5.1",
    "matplotlib>=3.10.3",
    "nltk>=3.9.1",
    "pandas>=2.3.0",
    "scikit-learn>=1.7.0",
    "seaborn>=0.13.2",
    "torch>=2.7.1",
    "transformers>=4.53.0",
    "tokenizers>=0.19.0",
]
```

---

## 📊 Veri Seti / Dataset

### Veri Kaynağı / Data Source
- **Kaynak**: E-ticaret ürün verileri
- **Boyut**: 3000 kayıt, 9 sütun
- **Format**: CSV

### Veri Sütunları / Data Columns
| Sütun / Column | Tip / Type | Açıklama / Description |
|----------------|------------|------------------------|
| `event_time` | object | Olay zamanı / Event timestamp |
| `event_type` | object | Olay tipi / Event type |
| `product_id` | int64 | Ürün ID'si / Product ID |
| `category_id` | int64 | Kategori ID'si / Category ID |
| `category_code` | object | Kategori kodu / Category code |
| `brand` | object | Marka / Brand |
| `price` | float64 | Fiyat / Price |
| `user_id` | int64 | Kullanıcı ID'si / User ID |
| `user_session` | object | Kullanıcı oturumu / User session |

### Veri İstatistikleri / Data Statistics
- **Toplam kayıt**: 3,000
- **Benzersiz kategori**: 99 (temizleme sonrası 71)
- **Benzersiz marka**: 481
- **Ortalama fiyat**: $292.99
- **Eksik değerler**: %31.7 category_code, %13.8 brand



## 📈 Proje Aşamaları / Project Phases

### 1. 📊 Keşifsel Veri Analizi (EDA)
- **Kategori analizi**: 99 benzersiz kategori tespit edildi
- **Marka analizi**: Samsung, Apple, Xiaomi en popüler markalar
- **Fiyat analizi**: Ortalama $292.99, maksimum $2,572.23
- **Görselleştirmeler**: Bar charts, histograms, box plots

### 2. 🔧 Veri Ön İşleme
- **Eksik değer temizliği**: 951 kayıt kaldırıldı
- **Kategori filtreleme**: Minimum 3 örnek filtresi
- **Metin temizleme**: Noktalama işaretleri, küçük harfe çevirme
- **Label Encoding**: 71 kategori → sayısal değerler

### 3. 🔤 TF-IDF Vektörleştirme
- **173 özellik** oluşturuldu
- **N-gram range**: (1, 2) - Unigram ve bigram
- **Stop words**: İngilizce stop words kaldırıldı
- **Sparsity**: %97.81 (çok seyrek matris)

### 4. 🤖 Machine Learning Modelleri
- **Logistic Regression**: %98.51 accuracy (en iyi)
- **Random Forest**: %90.05 accuracy
- **Naive Bayes**: %87.31 accuracy
- **Cross-validation**: 5-fold CV ile doğrulama

### 5. 🎨 Generative AI (GenAI)
- **HuggingFace T5-small** modeli
- **Pipeline**: text2text-generation
- **10 test ürünü** için açıklama üretimi
- **Template-based fallback** sistemi

### 6. 📊 Model Değerlendirme
- **Classification Report**: Tüm kategorilerde %100 precision/recall
- **Confusion Matrix**: En popüler 10 kategori için görselleştirme
- **Feature Importance**: Random Forest için özellik önem analizi
- **Örnek tahminler**: 10 test örneği ile doğrulama

---

## 🏆 Sonuçlar / Results

### Model Performansı / Model Performance
| Model | Accuracy | F1-Score | CV Mean | CV Std |
|-------|----------|----------|---------|--------|
| **Logistic Regression** | **0.9851** | **0.9803** | **0.9807** | **0.0087** |
| Random Forest | 0.9005 | 0.8575 | 0.9055 | 0.0150 |
| Naive Bayes | 0.8731 | 0.8237 | 0.8731 | 0.0037 |

### En İyi Model Detayları / Best Model Details
- **Model**: Logistic Regression
- **Test Accuracy**: 98.51%
- **F1-Score**: 98.03%
- **Cross-validation**: 98.07% (±1.73%)
- **Kategori sayısı**: 71
- **TF-IDF özellik sayısı**: 173

### GenAI Sonuçları / GenAI Results
- **Model**: T5-small (HuggingFace)
- **Test edilen ürün**: 10 farklı kategori
- **Başarı oranı**: %100 (template fallback ile)
- **Ortalama açıklama uzunluğu**: 70-80 karakter

---

## 📁 Dosya Açıklamaları / File Descriptions

### Modüler Yaklaşım / Modular Approach

#### `src/data_loader.py`
```python
class EcommerceDataLoader:
    """E-ticaret veri setini yüklemek için sınıf"""
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """CSV dosyasından veri yükle"""
    
    def get_data_info(self) -> dict:
        """Veri seti hakkında bilgi döndür"""
```

#### `src/preprocessing.py`
```python
class EcommercePreprocessor:
    """Veri ön işleme sınıfı"""
    
    def clean_text(self, text: str) -> str:
        """Metin temizleme"""
    
    def preprocess_for_ml(self, df: pd.DataFrame) -> tuple:
        """ML için veri hazırlama"""
```

#### `src/ml_models.py`
```python
class CategoryPredictor:
    """Kategori tahmin sınıfı"""
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Model eğitimi"""
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Tahmin yapma"""
```

#### `src/genai_utils.py`
```python
class ProductDescriptionGenerator:
    """Ürün açıklaması üretim sınıfı"""
    
    def generate_description(self, product_title: str) -> str:
        """Açıklama üretimi"""
```

### Tek Notebook / Single Notebook

#### `single_notebook/main_project.ipynb`
- **Cell 0**: Import'lar ve kurulum
- **Cell 1**: Veri yükleme ve ilk inceleme
- **Cell 2**: Keşifsel veri analizi (EDA)
- **Cell 3**: Veri ön işleme ve temizlik
- **Cell 4**: TF-IDF vektörleştirme
- **Cell 5**: Machine Learning modelleri
- **Cell 6**: Generative AI açıklama üretimi
- **Cell 7**: En iyi model detaylı analizi
- **Cell 8**: Model kaydetme ve sonuçlar

---

## 🎨 Görselleştirmeler / Visualizations

### EDA Görselleştirmeleri / EDA Visualizations
1. **Kategori dağılımı**: En popüler 15 kategori
2. **Marka analizi**: En popüler 10 marka
3. **Fiyat dağılımı**: Histogram, box plot, log dağılım
4. **Eksik değer analizi**: Bar chart

### ML Görselleştirmeleri / ML Visualizations
1. **Model karşılaştırması**: Accuracy, F1-score, CV mean
2. **Confusion Matrix**: En popüler 10 kategori
3. **Feature Importance**: Random Forest için özellik önem analizi
4. **TF-IDF analizi**: Kelime frekansı, değer dağılımı

### GenAI Görselleştirmeleri / GenAI Visualizations
1. **Açıklama uzunlukları**: Generated vs Template
2. **Başarı oranı**: Pasta grafiği
3. **Test edilen ürünler**: Kategori dağılımı

---

## 🔍 Teknik Detaylar / Technical Details

### TF-IDF Parametreleri / TF-IDF Parameters
```python
TfidfVectorizer(
    max_features=1000,      # En fazla 1000 özellik
    stop_words='english',   # İngilizce stop words
    ngram_range=(1, 2),     # Unigram ve bigram
    min_df=2,               # En az 2 dokümanda geçmeli
    max_df=0.95,            # %95'ten fazla dokümanda geçmemeli
    lowercase=True,         # Küçük harfe çevir
    strip_accents='unicode' # Aksanları kaldır
)
```

### ML Model Parametreleri / ML Model Parameters
```python
# Logistic Regression
LogisticRegression(
    random_state=42, 
    max_iter=1000,
    C=1.0
)

# Random Forest
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    max_depth=10
)

# Naive Bayes
MultinomialNB(alpha=1.0)
```

### GenAI Model Parametreleri / GenAI Model Parameters
```python
pipeline(
    "text2text-generation", 
    model="t5-small",
    max_length=100,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)
```

---

## 📊 Performans Metrikleri / Performance Metrics

### Accuracy (Doğruluk)
- **Logistic Regression**: 98.51%
- **Random Forest**: 90.05%
- **Naive Bayes**: 87.31%

### F1-Score
- **Logistic Regression**: 98.03%
- **Random Forest**: 85.75%
- **Naive Bayes**: 82.37%

### Cross-Validation
- **Logistic Regression**: 98.07% (±1.73%)
- **Random Forest**: 90.55% (±3.00%)
- **Naive Bayes**: 87.31% (±0.73%)

---

## 🚀 Gelecek Geliştirmeler / Future Improvements

### Kısa Vadeli / Short-term
1. **Model Fine-tuning**: Daha iyi hiperparametre optimizasyonu
2. **Feature Engineering**: Ek özellikler (fiyat, marka encoding)
3. **Ensemble Methods**: Voting, stacking modelleri
4. **Hyperparameter Tuning**: GridSearch, RandomizedSearch

### Orta Vadeli / Medium-term
1. **Deep Learning**: BERT, RoBERTa modelleri
2. **Multi-label Classification**: Çoklu kategori tahmini
3. **Real-time Prediction**: API geliştirme
4. **Model Monitoring**: Performans takibi

### Uzun Vadeli / Long-term
1. **Production Deployment**: Web uygulaması
2. **AutoML**: Otomatik model seçimi
3. **A/B Testing**: Model karşılaştırma
4. **Scalability**: Büyük veri setleri için optimizasyon

---

## 🐛 Bilinen Sorunlar / Known Issues

### GenAI Kalite Sorunu
- **Sorun**: T5-small modeli fine-tuning olmadan kalitesiz sonuçlar üretiyor
- **Çözüm**: Template-based fallback sistemi uygulandı
- **Gelecek**: Model fine-tuning veya daha büyük model kullanımı

### Overfitting Riski
- **Sorun**: %98.51 accuracy çok yüksek olabilir
- **Çözüm**: Cross-validation ile doğrulama
- **Gelecek**: Daha büyük test seti, regularization

---

## 📝 Katkıda Bulunma / Contributing

### Katkı Süreci / Contribution Process
1. Fork yapın / Fork the repository
2. Feature branch oluşturun / Create feature branch
3. Değişikliklerinizi commit edin / Commit your changes
4. Pull request gönderin / Submit pull request

### Kod Standartları / Code Standards
- **Python**: PEP 8 standartları
- **Docstrings**: Google style docstrings
- **Type Hints**: Python type annotations

---

## 📄 Lisans / License

Bu proje MIT lisansı altında lisanslanmıştır.

This project is licensed under the MIT License.

---

## 👥 Yazar / Author

**Harun Emirhan BOSTANCI** - E-ticaret Ürün Zekası Projesi

**Contact**: [Linkedin Profile](https://www.linkedin.com/in/haremir826/)

---


## 📞 İletişim / Contact

- **GitHub**: [Repository Link](https://github.com/haremir/product-intelligence-hub)
- **Email**: harunemirhan826@gmail.com
- **LinkedIn**: [LinkedIn Profile](https://www.linkedin.com/in/haremir826/)

---

## 🏷️ Etiketler / Tags

`machine-learning` `nlp` `e-commerce` `product-analysis` `category-prediction` `generative-ai` `tf-idf` `logistic-regression` `random-forest` `huggingface` `transformers` `t5` `python` `jupyter` `pandas` `scikit-learn` `matplotlib` `seaborn`

---

**⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın! / Don't forget to star this project if you liked it! ⭐**
