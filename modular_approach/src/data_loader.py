"""
Data Loader Module

Bu modül, e-ticaret veri setini yüklemek, train-test split yapmak
ve ML modelleri ile GenAI görevleri için veriyi hazırlamak için kullanılır.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EcommerceDataLoader:
    """
    E-ticaret veri seti için veri yükleme ve hazırlama sınıfı.
    """
    
    def __init__(self, data_path: str = "data/processed/sample_3k.csv"):
        """
        DataLoader'ı başlat.
        
        Args:
            data_path (str): İşlenmiş veri setinin yolu
        """
        self.data_path = Path(data_path)
        self.df = None
        self.label_encoder = LabelEncoder()
        self.category_mapping = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Veri setini yükle ve temel kontrolleri yap.
        
        Returns:
            pd.DataFrame: Yüklenen veri seti
        """
        try:
            logger.info(f"Veri seti yükleniyor: {self.data_path}")
            
            # Veri setini yükle
            self.df = pd.read_csv(self.data_path)
            
            logger.info(f"Veri seti başarıyla yüklendi: {self.df.shape}")
            logger.info(f"Sütunlar: {list(self.df.columns)}")
            
            # Temel veri kontrolü
            self._basic_data_check()
            
            return self.df
            
        except FileNotFoundError:
            logger.error(f"Veri dosyası bulunamadı: {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"Veri yükleme hatası: {e}")
            raise
    
    def _basic_data_check(self) -> None:
        """
        Temel veri kalitesi kontrollerini yap.
        """
        logger.info("Temel veri kontrolleri yapılıyor...")
        
        # Eksik değer kontrolü
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        logger.info("Eksik değer analizi:")
        for col, missing_count in missing_data.items():
            if missing_count > 0:
                logger.info(f"  {col}: {missing_count} ({missing_percent[col]:.1f}%)")
        
        # Duplicate kontrolü
        duplicate_count = self.df.duplicated().sum()
        logger.info(f"Tekrarlanan kayıt: {duplicate_count}")
        
        # Kategori sayısı kontrolü
        if 'category_code' in self.df.columns:
            unique_categories = self.df['category_code'].nunique()
            logger.info(f"Benzersiz kategori sayısı: {unique_categories}")
        
        # Marka sayısı kontrolü
        if 'brand' in self.df.columns:
            unique_brands = self.df['brand'].nunique()
            logger.info(f"Benzersiz marka sayısı: {unique_brands}")
    
    def prepare_for_ml(self, 
                      text_column: str = 'product_title',
                      category_column: str = 'category_code',
                      test_size: float = 0.2,
                      random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        ML modelleri için veriyi hazırla.
        
        Args:
            text_column (str): Metin sütunu adı
            category_column (str): Kategori sütunu adı
            test_size (float): Test seti oranı
            random_state (int): Random seed
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        logger.info("ML modelleri için veri hazırlanıyor...")
        
        if self.df is None:
            self.load_data()
        
        # Gerekli sütunları kontrol et
        required_columns = [text_column, category_column]
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            raise ValueError(f"Eksik sütunlar: {missing_columns}")
        
        # NaN değerleri olan satırları temizle
        df_clean = self.df.dropna(subset=[text_column, category_column]).copy()
        logger.info(f"Temizlik sonrası veri boyutu: {df_clean.shape}")
        
        # Kategori encoding
        df_clean, category_mapping = self._encode_categories(df_clean, category_column)
        self.category_mapping = category_mapping
        
        # Train-test split
        X = df_clean[text_column]
        y = df_clean[f'{category_column}_encoded']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Train seti boyutu: {len(X_train)}")
        logger.info(f"Test seti boyutu: {len(X_test)}")
        logger.info(f"Kategori sayısı: {len(category_mapping)}")
        
        return X_train, X_test, y_train, y_test
    
    def prepare_for_genai(self, 
                         text_column: str = 'product_title',
                         description_column: str = 'product_description',
                         sample_size: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        GenAI modelleri için veriyi hazırla.
        
        Args:
            text_column (str): Giriş metni sütunu (başlık)
            description_column (str): Hedef metin sütunu (açıklama)
            sample_size (int): Örnek sayısı
            
        Returns:
            Tuple: (input_texts, target_texts)
        """
        logger.info("GenAI modelleri için veri hazırlanıyor...")
        
        if self.df is None:
            self.load_data()
        
        # Gerekli sütunları kontrol et
        required_columns = [text_column]
        if description_column in self.df.columns:
            required_columns.append(description_column)
        
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Eksik sütunlar: {missing_columns}")
        
        # NaN değerleri olan satırları temizle
        df_clean = self.df.dropna(subset=[text_column]).copy()
        
        # Örnek veri seç
        if sample_size and sample_size < len(df_clean):
            df_sample = df_clean.sample(n=sample_size, random_state=42)
        else:
            df_sample = df_clean
        
        # Giriş ve hedef metinleri hazırla
        input_texts = df_sample[text_column]
        
        if description_column in df_sample.columns:
            target_texts = df_sample[description_column]
        else:
            # Eğer açıklama sütunu yoksa, başlığı tekrar kullan
            target_texts = df_sample[text_column]
            logger.warning(f"'{description_column}' sütunu bulunamadı, başlık kullanılıyor")
        
        logger.info(f"GenAI için {len(input_texts)} örnek hazırlandı")
        
        return input_texts, target_texts
    
    def _encode_categories(self, df: pd.DataFrame, category_column: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Kategorileri encode et.
        
        Args:
            df (pd.DataFrame): Veri seti
            category_column (str): Kategori sütunu
            
        Returns:
            Tuple: (encoded_df, category_mapping)
        """
        # Kategori sayısını azalt (çok fazla kategori var)
        category_counts = df[category_column].value_counts()
        
        # En az 5 ürünü olan kategorileri al
        valid_categories = category_counts[category_counts >= 5].index
        df_filtered = df[df[category_column].isin(valid_categories)].copy()
        
        logger.info(f"Kategori filtreleme: {len(category_counts)} → {len(valid_categories)}")
        
        # Label encoding
        df_filtered[f'{category_column}_encoded'] = self.label_encoder.fit_transform(
            df_filtered[category_column]
        )
        
        # Kategori mapping'i oluştur
        category_mapping = dict(zip(
            self.label_encoder.classes_, 
            self.label_encoder.transform(self.label_encoder.classes_)
        ))
        
        return df_filtered, category_mapping
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Veri seti özetini döndür.
        
        Returns:
            Dict: Veri seti özeti
        """
        if self.df is None:
            self.load_data()
        
        summary = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum()
        }
        
        # Kategori bilgisi
        if 'category_code' in self.df.columns:
            summary['unique_categories'] = self.df['category_code'].nunique()
            summary['top_categories'] = self.df['category_code'].value_counts().head(5).to_dict()
        
        # Marka bilgisi
        if 'brand' in self.df.columns:
            summary['unique_brands'] = self.df['brand'].nunique()
            summary['top_brands'] = self.df['brand'].value_counts().head(5).to_dict()
        
        # Fiyat bilgisi
        if 'price' in self.df.columns:
            summary['price_stats'] = {
                'min': self.df['price'].min(),
                'max': self.df['price'].max(),
                'mean': self.df['price'].mean(),
                'median': self.df['price'].median()
            }
        
        return summary
    
    def save_processed_data(self, output_path: str = "modular_approach/data/processed/ml_ready_data.csv") -> None:
        """
        İşlenmiş veriyi kaydet.
        
        Args:
            output_path (str): Çıktı dosya yolu
        """
        if self.df is None:
            logger.warning("Kaydedilecek veri yok!")
            return
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.df.to_csv(output_path, index=False)
            logger.info(f"İşlenmiş veri kaydedildi: {output_path}")
            
        except Exception as e:
            logger.error(f"Veri kaydetme hatası: {e}")
            raise


# Notebook'larda kullanım için basit test fonksiyonu
def test_data_loader():
    """
    DataLoader'ı test et (notebook'larda kullanım için).
    """
    loader = EcommerceDataLoader()
    df = loader.load_data()
    print(f"✅ Veri yüklendi: {df.shape}")
    return loader, df
