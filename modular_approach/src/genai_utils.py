"""
GenAI Utilities for Product Description Generation
Ürün açıklaması üretimi için GenAI araçları
"""

import pandas as pd
import numpy as np
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import logging
import warnings
import os
import json
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductDescriptionDataset(Dataset):
    """
    T5 modeli için özel dataset sınıfı
    """
    
    def __init__(self, input_texts, target_texts, tokenizer, max_length=128):
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.input_texts)
    
    def __getitem__(self, idx):
        input_text = str(self.input_texts.iloc[idx])
        target_text = str(self.target_texts.iloc[idx])
        
        # Input encoding
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Target encoding
        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }


class ProductDescriptionGenerator:
    """
    T5 modeli kullanarak ürün açıklaması üreten sınıf
    """
    
    def __init__(self, model_name='t5-small', device=None):
        """
        Args:
            model_name (str): T5 model adı ('t5-small', 't5-base', 't5-large')
            device (str): GPU/CPU cihazı
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"T5 modeli yükleniyor: {model_name}")
        logger.info(f"Cihaz: {self.device}")
        
        # Tokenizer ve model yükle
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Modeli cihaza taşı
        self.model.to(self.device)
        
        # Eğitim durumu
        self.is_trained = False
        self.training_history = []
        
    def prepare_data(self, data_path, test_size=0.2):
        """
        GenAI için veriyi hazırla
        
        Args:
            data_path (str): GenAI veri dosyası yolu
            test_size (float): Test seti oranı
            
        Returns:
            tuple: train_dataset, test_dataset
        """
        logger.info(f"GenAI verisi yükleniyor: {data_path}")
        
        # Veriyi yükle
        df = pd.read_csv(data_path)
        
        input_texts = df['input_text']
        target_texts = df['target_text']
        
        logger.info(f"Veri boyutu: {len(df)}")
        logger.info(f"Örnek input: {input_texts.iloc[0]}")
        logger.info(f"Örnek target: {target_texts.iloc[0]}")
        
        # Train-test split
        train_inputs, test_inputs, train_targets, test_targets = train_test_split(
            input_texts, target_texts, test_size=test_size, random_state=42
        )
        
        # Dataset'leri oluştur
        train_dataset = ProductDescriptionDataset(
            train_inputs, train_targets, self.tokenizer
        )
        test_dataset = ProductDescriptionDataset(
            test_inputs, test_targets, self.tokenizer
        )
        
        logger.info(f"Train boyutu: {len(train_dataset)}")
        logger.info(f"Test boyutu: {len(test_dataset)}")
        
        return train_dataset, test_dataset
    
    def train_model(self, train_dataset, test_dataset=None, 
                   epochs=3, batch_size=4, learning_rate=5e-5):
        """
        T5 modelini fine-tune et
        
        Args:
            train_dataset: Eğitim dataset'i
            test_dataset: Test dataset'i
            epochs: Epoch sayısı
            batch_size: Batch boyutu
            learning_rate: Öğrenme oranı
        """
        logger.info("T5 modeli eğitiliyor...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./t5_product_description",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="steps" if test_dataset else "no",
            eval_steps=100 if test_dataset else None,
            save_steps=1000,
            save_total_limit=2,
            load_best_model_at_end=True if test_dataset else False,
            metric_for_best_model="eval_loss" if test_dataset else None,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Eğitim
        logger.info("Eğitim başlıyor...")
        train_result = trainer.train()
        
        # Eğitim geçmişini kaydet
        self.training_history = trainer.state.log_history
        self.is_trained = True
        
        logger.info("✅ Model eğitimi tamamlandı!")
        logger.info(f"Eğitim loss: {train_result.training_loss:.4f}")
        
        return train_result
    
    def generate_description(self, input_text, max_length=128, num_beams=4):
        """
        Verilen input için açıklama üret
        
        Args:
            input_text (str): Giriş metni (ürün başlığı/kategorisi)
            max_length (int): Maksimum çıktı uzunluğu
            num_beams (int): Beam search parametresi
            
        Returns:
            str: Üretilen açıklama
        """
        if not self.is_trained:
            logger.warning("Model henüz eğitilmemiş! Varsayılan model kullanılıyor.")
        
        # Input'u tokenize et
        inputs = self.tokenizer.encode(
            input_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=128
        ).to(self.device)
        
        # Açıklama üret
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Token'ları metne çevir
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def batch_generate_descriptions(self, input_texts, max_length=128, num_beams=4):
        """
        Birden fazla input için açıklama üret
        
        Args:
            input_texts (list): Giriş metinleri listesi
            max_length (int): Maksimum çıktı uzunluğu
            num_beams (int): Beam search parametresi
            
        Returns:
            list: Üretilen açıklamalar listesi
        """
        logger.info(f"{len(input_texts)} açıklama üretiliyor...")
        
        generated_descriptions = []
        
        for i, input_text in enumerate(input_texts):
            try:
                description = self.generate_description(
                    input_text, max_length, num_beams
                )
                generated_descriptions.append(description)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"İlerleme: {i + 1}/{len(input_texts)}")
                    
            except Exception as e:
                logger.error(f"Açıklama üretilirken hata: {e}")
                generated_descriptions.append("Açıklama üretilemedi")
        
        return generated_descriptions
    
    def evaluate_model(self, test_dataset, num_samples=10):
        """
        Model performansını değerlendir
        
        Args:
            test_dataset: Test dataset'i
            num_samples: Test edilecek örnek sayısı
        """
        logger.info("Model performansı değerlendiriliyor...")
        
        # Test örneklerini al
        test_samples = min(num_samples, len(test_dataset))
        
        results = []
        
        for i in range(test_samples):
            sample = test_dataset[i]
            
            # Input'u decode et
            input_text = self.tokenizer.decode(
                sample['input_ids'], 
                skip_special_tokens=True
            )
            
            # Gerçek target'ı decode et
            target_text = self.tokenizer.decode(
                sample['labels'], 
                skip_special_tokens=True
            )
            
            # Tahmin üret
            predicted_text = self.generate_description(input_text)
            
            results.append({
                'input': input_text,
                'target': target_text,
                'predicted': predicted_text,
                'correct': target_text.lower() in predicted_text.lower()
            })
        
        # Sonuçları analiz et
        correct_predictions = sum(1 for r in results if r['correct'])
        accuracy = correct_predictions / len(results)
        
        logger.info(f"Test Accuracy: {accuracy:.4f} ({correct_predictions}/{len(results)})")
        
        return results, accuracy
    
    def save_model(self, model_path):
        """
        Eğitilmiş modeli kaydet
        
        Args:
            model_path (str): Model kayıt yolu
        """
        if not self.is_trained:
            logger.warning("Model henüz eğitilmemiş!")
            return
        
        # Model klasörü oluştur
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Modeli kaydet
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # Eğitim bilgilerini kaydet
        training_info = {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'device': self.device
        }
        
        info_path = os.path.join(model_path, 'training_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(training_info, f, indent=2, default=str)
        
        logger.info(f"Model kaydedildi: {model_path}")
    
    def load_model(self, model_path):
        """
        Kaydedilmiş modeli yükle
        
        Args:
            model_path (str): Model dosyası yolu
        """
        logger.info(f"Model yükleniyor: {model_path}")
        
        # Modeli yükle
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        
        # Modeli cihaza taşı
        self.model.to(self.device)
        
        # Eğitim bilgilerini yükle
        info_path = os.path.join(model_path, 'training_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                training_info = json.load(f)
                self.is_trained = training_info.get('is_trained', False)
                self.training_history = training_info.get('training_history', [])
        
        logger.info("Model başarıyla yüklendi!")


def test_genai_generator():
    """
    ProductDescriptionGenerator sınıfını test et
    """
    print("🧪 ProductDescriptionGenerator Test Ediliyor...")
    
    # Test verisi oluştur
    test_data = pd.DataFrame({
        'input_text': [
            'electronics smartphone',
            'apparel shoes',
            'computers laptop'
        ],
        'target_text': [
            'High-quality smartphone with advanced features',
            'Comfortable athletic shoes for daily use',
            'Powerful laptop for professional work'
        ]
    })
    
    # Test dosyası oluştur
    test_path = '../../data/processed/test_genai_data.csv'
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    test_data.to_csv(test_path, index=False)
    
    # Generator oluştur
    generator = ProductDescriptionGenerator(model_name='t5-small')
    
    # Veriyi hazırla
    train_dataset, test_dataset = generator.prepare_data(test_path)
    
    print("✅ ProductDescriptionGenerator test başarılı!")
    print(f"Train dataset boyutu: {len(train_dataset)}")
    print(f"Test dataset boyutu: {len(test_dataset)}")
    
    # Test dosyasını sil
    os.remove(test_path)
    
    return generator


if __name__ == "__main__":
    test_genai_generator()

