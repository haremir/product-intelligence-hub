"""
Preprocessing Module

Bu modül, e-ticaret veri seti için metin temizleme, özellik mühendisliği
ve veri ön işleme fonksiyonlarını içerir.
"""

import pandas as pd
import numpy as np
import re
import string
from typing import List, Optional, Tuple, Dict, Any
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Metin ön işleme sınıfı.
    """
    
    def __init__(self, language: str = 'english'):
        """
        TextPreprocessor'ı başlat.
        
        Args:
            language (str): Dil (stopwords için)
        """
        self.language = language
        self.stop_words = set()
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # NLTK kaynaklarını indir (eğer yoksa)
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        # Stopwords'ü yükle
        self.stop_words = set(stopwords.words(language))
        
        # Türkçe stopwords ekle (e-ticaret verisi için)
        turkish_stopwords = {
            've', 'ile', 'için', 'bu', 'bir', 'da', 'de', 'mi', 'mu', 'mü',
            'ama', 'fakat', 'lakin', 'ancak', 'sadece', 'yalnız', 'hem',
            'ya', 'veya', 'yoksa', 'eğer', 'ise', 'çünkü', 'zira', 'madem',
            'ne', 'nasıl', 'nerede', 'nereden', 'nereye', 'kim', 'hangi',
            'kaç', 'kaçıncı', 'ne kadar', 'ne zaman', 'niçin', 'niye'
        }
        self.stop_words.update(turkish_stopwords)
    
    def clean_text(self, text: str, 
                   remove_punctuation: bool = True,
                   remove_numbers: bool = False,
                   remove_stopwords: bool = True,
                   lemmatize: bool = True,
                   stem: bool = False,
                   min_length: int = 2) -> str:
        """
        Metni temizle ve normalize et.
        
        Args:
            text (str): Temizlenecek metin
            remove_punctuation (bool): Noktalama işaretlerini kaldır
            remove_numbers (bool): Sayıları kaldır
            remove_stopwords (bool): Stopwords'leri kaldır
            lemmatize (bool): Lemmatization uygula
            stem (bool): Stemming uygula
            min_length (int): Minimum kelime uzunluğu
            
        Returns:
            str: Temizlenmiş metin
        """
        if pd.isna(text) or text == '':
            return ''
        
        # String'e çevir
        text = str(text).lower().strip()
        
        # Noktalama işaretlerini kaldır
        if remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Sayıları kaldır
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Fazla boşlukları temizle
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Kelimelere ayır
        words = text.split()
        
        # Stopwords'leri kaldır
        if remove_stopwords:
            words = [word for word in words if word not in self.stop_words]
        
        # Minimum uzunluk kontrolü
        words = [word for word in words if len(word) >= min_length]
        
        # Lemmatization
        if lemmatize:
            words = [self.lemmatizer.lemmatize(word) for word in words]
        
        # Stemming (lemmatization'dan sonra)
        if stem:
            words = [self.stemmer.stem(word) for word in words]
        
        return ' '.join(words)
    
    def clean_text_series(self, text_series: pd.Series, **kwargs) -> pd.Series:
        """
        Pandas Series'deki metinleri toplu olarak temizle.
        
        Args:
            text_series (pd.Series): Temizlenecek metin serisi
            **kwargs: clean_text fonksiyonuna geçirilecek parametreler
            
        Returns:
            pd.Series: Temizlenmiş metin serisi
        """
        logger.info(f"{len(text_series)} metin temizleniyor...")
        
        cleaned_series = text_series.apply(lambda x: self.clean_text(x, **kwargs))
        
        # Boş metinleri say
        empty_count = (cleaned_series == '').sum()
        logger.info(f"Temizlik tamamlandı. Boş metin sayısı: {empty_count}")
        
        return cleaned_series


class FeatureEngineer:
    """
    Özellik mühendisliği sınıfı.
    """
    
    def __init__(self):
        """
        FeatureEngineer'ı başlat.
        """
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
    
    def extract_text_features(self, text_series: pd.Series) -> pd.DataFrame:
        """
        Metin özelliklerini çıkar.
        
        Args:
            text_series (pd.Series): Metin serisi
            
        Returns:
            pd.DataFrame: Özellik matrisi
        """
        logger.info("Metin özellikleri çıkarılıyor...")
        
        features = pd.DataFrame()
        
        # Temel metin özellikleri
        features['text_length'] = text_series.str.len()
        features['word_count'] = text_series.str.split().str.len()
        features['avg_word_length'] = text_series.str.split().apply(
            lambda x: np.mean([len(word) for word in x]) if x else 0
        )
        features['unique_word_ratio'] = text_series.str.split().apply(
            lambda x: len(set(x)) / len(x) if x else 0
        )
        
        # Büyük harf oranı
        features['uppercase_ratio'] = text_series.apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if str(x) else 0
        )
        
        # Sayı oranı
        features['digit_ratio'] = text_series.apply(
            lambda x: sum(1 for c in str(x) if c.isdigit()) / len(str(x)) if str(x) else 0
        )
        
        # Noktalama oranı
        features['punctuation_ratio'] = text_series.apply(
            lambda x: sum(1 for c in str(x) if c in string.punctuation) / len(str(x)) if str(x) else 0
        )
        
        logger.info(f"Metin özellikleri çıkarıldı: {features.shape}")
        
        return features
    
    def create_tfidf_features(self, text_series: pd.Series, 
                            max_features: int = 1000,
                            ngram_range: Tuple[int, int] = (1, 2)) -> pd.DataFrame:
        """
        TF-IDF özelliklerini oluştur.
        
        Args:
            text_series (pd.Series): Metin serisi
            max_features (int): Maksimum özellik sayısı
            ngram_range (Tuple): N-gram aralığı
            
        Returns:
            pd.DataFrame: TF-IDF özellik matrisi
        """
        logger.info("TF-IDF özellikleri oluşturuluyor...")
        
        # TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        # TF-IDF matrisini oluştur
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_series)
        
        # DataFrame'e çevir
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=feature_names
        )
        
        logger.info(f"TF-IDF özellikleri oluşturuldu: {tfidf_df.shape}")
        
        return tfidf_df
    
    def create_brand_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Marka özelliklerini oluştur.
        
        Args:
            df (pd.DataFrame): Veri seti
            
        Returns:
            pd.DataFrame: Marka özellikleri
        """
        if 'brand' not in df.columns:
            logger.warning("'brand' sütunu bulunamadı!")
            return pd.DataFrame()
        
        logger.info("Marka özellikleri oluşturuluyor...")
        
        brand_features = pd.DataFrame()
        
        # Marka popülerliği (ürün sayısı)
        brand_counts = df['brand'].value_counts()
        brand_features['brand_popularity'] = df['brand'].map(brand_counts)
        
        # Marka kategorisi (popüler/orta/az popüler)
        brand_features['brand_category'] = pd.cut(
            brand_features['brand_popularity'],
            bins=[0, 10, 50, float('inf')],
            labels=['low', 'medium', 'high']
        )
        
        # Marka encoding - categorical hatası düzeltildi
        brand_encoder = LabelEncoder()
        brand_features['brand_encoded'] = brand_encoder.fit_transform(df['brand'].fillna('unknown'))
        
        logger.info(f"Marka özellikleri oluşturuldu: {brand_features.shape}")
        
        return brand_features
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fiyat özelliklerini oluştur.
        
        Args:
            df (pd.DataFrame): Veri seti
            
        Returns:
            pd.DataFrame: Fiyat özellikleri
        """
        if 'price' not in df.columns:
            logger.warning("'price' sütunu bulunamadı!")
            return pd.DataFrame()
        
        logger.info("Fiyat özellikleri oluşturuluyor...")
        
        price_features = pd.DataFrame()
        
        # Fiyat kategorileri - NaN değerleri handle et
        price_bins = [0, 50, 100, 250, 500, 1000, float('inf')]
        price_labels = ['very_low', 'low', 'medium', 'high', 'very_high', 'luxury']
        
        # NaN değerleri 'unknown' olarak işaretle
        price_features['price_category'] = pd.cut(
            df['price'].fillna(-1),  # NaN'ları -1 ile doldur
            bins=price_bins,
            labels=price_labels,
            include_lowest=True
        )
        
        # NaN değerleri 'unknown' olarak değiştir
        price_features['price_category'] = price_features['price_category'].fillna('unknown')
        
        # Fiyat encoding - LabelEncoder kullan
        le = LabelEncoder()
        price_features['price_encoded'] = le.fit_transform(price_features['price_category'])
        
        # Log fiyat - NaN değerleri 0 olarak işle
        price_features['log_price'] = np.log1p(df['price'].fillna(0))
        
        # Fiyat normalize edilmiş - NaN değerleri 0 olarak işle
        price_normalized = self.scaler.fit_transform(df[['price']].fillna(0))
        price_features['price_normalized'] = price_normalized.flatten()
        
        logger.info(f"Fiyat özellikleri oluşturuldu: {price_features.shape}")
        
        return price_features


class EcommercePreprocessor:
    """
    E-ticaret veri seti için ana ön işleme sınıfı.
    """
    
    def __init__(self):
        """
        EcommercePreprocessor'ı başlat.
        """
        self.text_preprocessor = TextPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.preprocessing_config = {}
    
    def preprocess_for_ml(self, 
                         df: pd.DataFrame,
                         text_column: str = 'product_title',
                         category_column: str = 'category_code',
                         use_tfidf: bool = True,
                         use_brand_features: bool = True,
                         use_price_features: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        ML modelleri için veriyi ön işle.
        
        Args:
            df (pd.DataFrame): Veri seti
            text_column (str): Metin sütunu
            category_column (str): Kategori sütunu
            use_tfidf (bool): TF-IDF özellikleri kullan
            use_brand_features (bool): Marka özellikleri kullan
            use_price_features (bool): Fiyat özellikleri kullan
            
        Returns:
            Tuple: (X_features, y_target)
        """
        logger.info("ML modelleri için veri ön işleniyor...")
        
        # NaN değerleri temizle - sadece kategori sütunundan
        df_clean = df.dropna(subset=[category_column]).copy()
        logger.info(f"Temizlik sonrası veri boyutu: {df_clean.shape}")
        
        # Metin sütunu yoksa, kategori sütununu kullan
        if text_column not in df_clean.columns:
            text_column = category_column
            logger.info(f"'{text_column}' sütunu bulunamadı, '{category_column}' kullanılıyor")
        
        # Metin temizleme
        df_clean[f'{text_column}_cleaned'] = self.text_preprocessor.clean_text_series(
            df_clean[text_column],
            remove_punctuation=True,
            remove_numbers=False,
            remove_stopwords=True,
            lemmatize=True,
            stem=False
        )
        
        # Özellik matrisini oluştur
        feature_dfs = []
        
        # TF-IDF özellikleri
        if use_tfidf:
            tfidf_features = self.feature_engineer.create_tfidf_features(
                df_clean[f'{text_column}_cleaned']
            )
            feature_dfs.append(tfidf_features)
        
        # Metin özellikleri
        text_features = self.feature_engineer.extract_text_features(
            df_clean[f'{text_column}_cleaned']
        )
        feature_dfs.append(text_features)
        
        # Marka özellikleri
        if use_brand_features and 'brand' in df_clean.columns:
            brand_features = self.feature_engineer.create_brand_features(df_clean)
            if not brand_features.empty:
                feature_dfs.append(brand_features)
        
        # Fiyat özellikleri
        if use_price_features and 'price' in df_clean.columns:
            price_features = self.feature_engineer.create_price_features(df_clean)
            if not price_features.empty:
                feature_dfs.append(price_features)
        
        # Tüm özellikleri birleştir
        if feature_dfs:
            X_features = pd.concat(feature_dfs, axis=1)
        else:
            # Hiç özellik yoksa, basit bir özellik matrisi oluştur
            X_features = pd.DataFrame(index=df_clean.index)
            X_features['text_length'] = df_clean[f'{text_column}_cleaned'].str.len()
        
        # NaN değerleri doldur
        X_features = X_features.fillna(0)
        
        # Hedef değişken - aynı index'i kullan
        y_target = df_clean[category_column]
        
        # Boyut tutarlılığını kontrol et
        if X_features.shape[0] != y_target.shape[0]:
            logger.warning(f"Boyut tutarsızlığı: X_features {X_features.shape}, y_target {y_target.shape}")
            # Ortak index'leri bul
            common_index = X_features.index.intersection(y_target.index)
            X_features = X_features.loc[common_index]
            y_target = y_target.loc[common_index]
            logger.info(f"Boyut düzeltildi: X_features {X_features.shape}, y_target {y_target.shape}")
        
        logger.info(f"Ön işleme tamamlandı. Özellik boyutu: {X_features.shape}")
        
        return X_features, y_target
    
    def preprocess_for_genai(self, 
                           df: pd.DataFrame,
                           text_column: str = 'product_title',
                           description_column: str = 'product_description',
                           sample_size: Optional[int] = None) -> Tuple[pd.Series, pd.Series]:
        """
        GenAI modelleri için veriyi ön işle.
        
        Args:
            df (pd.DataFrame): Veri seti
            text_column (str): Giriş metni sütunu (kategori)
            description_column (str): Hedef metin sütunu (marka)
            sample_size (Optional[int]): Örnek sayısı
            
        Returns:
            Tuple: (input_texts, target_texts)
        """
        logger.info("GenAI modelleri için veri ön işleniyor...")
        
        # NaN değerleri temizle
        df_clean = df.dropna(subset=[text_column, description_column]).copy()
        
        # Örnek seç
        if sample_size and sample_size < len(df_clean):
            df_sample = df_clean.sample(n=sample_size, random_state=42)
        else:
            df_sample = df_clean
        
        # Giriş metinlerini temizle (kategori)
        input_texts = self.text_preprocessor.clean_text_series(
            df_sample[text_column],
            remove_punctuation=True,
            remove_numbers=False,
            remove_stopwords=False,  # GenAI için stopwords'leri koru
            lemmatize=False,  # GenAI için lemmatization yapma
            stem=False
        )
        
        # Hedef metinleri hazırla (marka)
        target_texts = df_sample[description_column]
        
        # GenAI için daha anlamlı hedef metinler oluştur
        # Kategori + Marka kombinasyonu ile ürün açıklaması oluştur
        enhanced_targets = []
        for idx, row in df_sample.iterrows():
            category = str(row[text_column]).replace('.', ' ').title()
            brand = str(row[description_column]).title()
            price = row.get('price', '')
            
            # Fiyat bilgisi varsa ekle
            if pd.notna(price) and price > 0:
                price_str = f" priced at ${price:.2f}"
            else:
                price_str = ""
            
            # Ürün açıklaması oluştur
            description = f"{brand} {category}{price_str}"
            enhanced_targets.append(description)
        
        target_texts = pd.Series(enhanced_targets, index=df_sample.index)
        
        logger.info(f"GenAI ön işleme tamamlandı. Örnek sayısı: {len(input_texts)}")
        
        return input_texts, target_texts
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Ön işleme özetini döndür.
        
        Returns:
            Dict: Ön işleme özeti
        """
        summary = {
            'text_preprocessor': {
                'language': self.text_preprocessor.language,
                'stop_words_count': len(self.text_preprocessor.stop_words)
            },
            'feature_engineer': {
                'tfidf_vectorizer': self.feature_engineer.tfidf_vectorizer is not None
            },
            'config': self.preprocessing_config
        }
        
        return summary


# Notebook'larda kullanım için basit test fonksiyonu
def test_preprocessor():
    """
    Preprocessor'ı test et (notebook'larda kullanım için).
    """
    preprocessor = EcommercePreprocessor()
    print("✅ Preprocessor başlatıldı")
    return preprocessor

