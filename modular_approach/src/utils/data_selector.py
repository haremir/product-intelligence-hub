import pandas as pd
import numpy as np
from pathlib import Path

def create_random_sample(input_file_path, output_file_path, sample_size=3000, random_state=42):
    """
    Büyük veri setinden rastgele örnek alır ve yeni dosyaya kaydeder.
    
    Parameters:
    -----------
    input_file_path : str
        Orijinal veri setinin dosya yolu
    output_file_path : str
        Oluşturulacak örnek veri setinin dosya yolu
    sample_size : int
        Örneklem boyutu (varsayılan: 3000)
    random_state : int
        Rastgelelik için seed değeri (tekrarlanabilir sonuçlar için)
    """
    
    print(f"Veri seti okunuyor: {input_file_path}")
    
    try:
        # Veri setini oku
        df = pd.read_csv(input_file_path)
        
        print(f"Orijinal veri seti boyutu: {len(df):,} satır")
        print(f"Sütunlar: {list(df.columns)}")
        
        # Eğer veri seti sample_size'dan küçükse uyarı ver
        if len(df) < sample_size:
            print(f"⚠️  Uyarı: Veri setinde sadece {len(df)} satır var, {sample_size} talep edildi.")
            sample_size = len(df)
            print(f"Tüm veri seti ({sample_size} satır) kullanılacak.")
        
        # Rastgele örnekleme yap
        np.random.seed(random_state)
        random_sample = df.sample(n=sample_size, random_state=random_state)
        
        # Çıktı dizinini oluştur
        output_path = Path(output_file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Örnek veri setini kaydet
        random_sample.to_csv(output_file_path, index=False)
        
        print(f"\n✅ Başarılı!")
        print(f"📁 {sample_size:,} satırlık rastgele örnek oluşturuldu")
        print(f"💾 Kaydedildi: {output_file_path}")
        
        # Özet istatistikler
        print(f"\n📊 Örnek Veri Seti Özeti:")
        print(f"   • Satır sayısı: {len(random_sample):,}")
        print(f"   • Sütun sayısı: {len(random_sample.columns)}")
        print(f"   • Dosya boyutu: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # İlk birkaç satırı göster
        print(f"\n🔍 İlk 3 satır:")
        print(random_sample.head(3).to_string())
        
        return random_sample
        
    except FileNotFoundError:
        print(f"❌ Hata: {input_file_path} dosyası bulunamadı!")
        return None
    except Exception as e:
        print(f"❌ Hata oluştu: {str(e)}")
        return None

# Kullanım örneği
if __name__ == "__main__":
    # Dosya yollarını belirtin
    INPUT_FILE = r"C:\Users\emirh\Desktop\product-intelligence-hub\data\raw\2019-Nov.csv"  # Büyük veri setinizin yolu
    OUTPUT_FILE = r"data/processed/sample_3k.csv"  # Oluşturulacak örnek dosya
    
    # Rastgele 3000 satırlık örnek oluştur
    sample_df = create_random_sample(
        input_file_path=INPUT_FILE,
        output_file_path=OUTPUT_FILE,
        sample_size=3000,
        random_state=42  # Tekrarlanabilir sonuçlar için
    )
