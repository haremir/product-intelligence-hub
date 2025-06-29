"""
ML Models for Category Prediction
Kategori tahmini için makine öğrenmesi modelleri
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import logging
import warnings
warnings.filterwarnings('ignore')

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CategoryPredictor:
    """
    E-ticaret ürün kategorilerini tahmin eden ML model sınıfı
    """
    
    def __init__(self, models=None):
        """
        Args:
            models (dict): Model isimleri ve sınıfları
        """
        self.models = models or {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(random_state=42, probability=True),
            'naive_bayes': MultinomialNB()
        }
        
        self.trained_models = {}
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.best_model_name = None
        
    def prepare_data(self, data_path):
        """
        ML için hazırlanmış veriyi yükle ve hazırla
        
        Args:
            data_path (str): ML veri dosyası yolu
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        logger.info(f"ML verisi yükleniyor: {data_path}")
        
        # Veriyi yükle
        df = pd.read_csv(data_path)
        
        # Hedef değişkeni ayır
        y = df['target']
        X = df.drop('target', axis=1)
        
        # Hedef değişkeni encode et
        y_encoded = self.label_encoder.fit_transform(y)
        
        logger.info(f"Veri boyutu: {X.shape}")
        logger.info(f"Kategori sayısı: {len(self.label_encoder.classes_)}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        logger.info(f"Train boyutu: {X_train.shape}")
        logger.info(f"Test boyutu: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, y_train, cv_folds=5):
        """
        Tüm modelleri eğit ve performanslarını karşılaştır
        
        Args:
            X_train: Eğitim özellikleri
            y_train: Eğitim hedefi
            cv_folds: Cross-validation fold sayısı
        """
        logger.info("Modeller eğitiliyor...")
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"{name} modeli eğitiliyor...")
            
            try:
                # Modeli eğit
                model.fit(X_train, y_train)
                self.trained_models[name] = model
                
                # Cross-validation skoru
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds)
                
                results[name] = {
                    'model': model,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores
                }
                
                logger.info(f"{name} - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                logger.error(f"{name} modeli eğitilirken hata: {e}")
                continue
        
        # En iyi modeli seç
        if results:
            best_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
            self.best_model = results[best_name]['model']
            self.best_model_name = best_name
            
            logger.info(f"En iyi model: {best_name} (CV Score: {results[best_name]['cv_mean']:.4f})")
        
        return results
    
    def evaluate_models(self, X_test, y_test):
        """
        Test seti üzerinde modelleri değerlendir
        
        Args:
            X_test: Test özellikleri
            y_test: Test hedefi
            
        Returns:
            dict: Model performans sonuçları
        """
        logger.info("Modeller test ediliyor...")
        
        results = {}
        
        for name, model in self.trained_models.items():
            try:
                # Tahminler
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Metrikler
                accuracy = accuracy_score(y_test, y_pred)
                
                results[name] = {
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                logger.info(f"{name} - Test Accuracy: {accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"{name} modeli test edilirken hata: {e}")
                continue
        
        return results
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='random_forest'):
        """
        En iyi model için hyperparameter tuning
        
        Args:
            X_train: Eğitim özellikleri
            y_train: Eğitim hedefi
            model_name: Tune edilecek model adı
        """
        logger.info(f"{model_name} için hyperparameter tuning...")
        
        if model_name == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42)
            
        elif model_name == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            model = GradientBoostingClassifier(random_state=42)
            
        else:
            logger.warning(f"{model_name} için hyperparameter tuning tanımlı değil")
            return None
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        logger.info(f"En iyi parametreler: {grid_search.best_params_}")
        logger.info(f"En iyi CV skoru: {grid_search.best_score_:.4f}")
        
        # En iyi modeli güncelle
        self.trained_models[model_name] = grid_search.best_estimator_
        self.best_model = grid_search.best_estimator_
        self.best_model_name = model_name
        
        return grid_search.best_estimator_
    
    def predict_category(self, text_features, model_name=None):
        """
        Yeni ürün için kategori tahmini
        
        Args:
            text_features: Ürün özellikleri (TF-IDF vektörü)
            model_name: Kullanılacak model adı (None ise en iyi model)
            
        Returns:
            tuple: (tahmin edilen kategori, güven skoru)
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.trained_models:
            raise ValueError(f"{model_name} modeli eğitilmemiş")
        
        model = self.trained_models[model_name]
        
        # Tahmin
        prediction = model.predict([text_features])[0]
        probability = model.predict_proba([text_features])[0].max() if hasattr(model, 'predict_proba') else None
        
        # Kategori adını geri çevir
        category_name = self.label_encoder.inverse_transform([prediction])[0]
        
        return category_name, probability
    
    def save_model(self, filepath, model_name=None):
        """
        Modeli kaydet
        
        Args:
            filepath: Kayıt yolu
            model_name: Kaydedilecek model adı
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.trained_models:
            raise ValueError(f"{model_name} modeli eğitilmemiş")
        
        model_data = {
            'model': self.trained_models[model_name],
            'label_encoder': self.label_encoder,
            'model_name': model_name
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model kaydedildi: {filepath}")
    
    def load_model(self, filepath):
        """
        Kaydedilmiş modeli yükle
        
        Args:
            filepath: Model dosyası yolu
        """
        model_data = joblib.load(filepath)
        
        self.trained_models[model_data['model_name']] = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        
        logger.info(f"Model yüklendi: {filepath}")


def test_category_predictor():
    """
    CategoryPredictor sınıfını test et
    """
    print("🧪 CategoryPredictor Test Ediliyor...")
    
    # Test verisi oluştur
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    
    X_test = np.random.rand(n_samples, n_features)
    y_test = np.random.choice(['electronics', 'apparel', 'books'], n_samples)
    
    # Label encoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_test)
    
    # Train-test split
    X_train, X_test_split, y_train, y_test_split = train_test_split(
        X_test, y_encoded, test_size=0.2, random_state=42
    )
    
    # Model oluştur ve eğit
    predictor = CategoryPredictor()
    results = predictor.train_models(X_train, y_train)
    
    # Test et
    test_results = predictor.evaluate_models(X_test_split, y_test_split)
    
    print("✅ CategoryPredictor test başarılı!")
    print(f"Eğitilen model sayısı: {len(predictor.trained_models)}")
    print(f"En iyi model: {predictor.best_model_name}")
    
    return predictor


if __name__ == "__main__":
    test_category_predictor()

