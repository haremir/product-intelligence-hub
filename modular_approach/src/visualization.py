"""
Veri Görselleştirme Modülü
E-ticaret veri seti için özelleştirilmiş görselleştirme fonksiyonları
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Türkçe font ayarları
plt.rcParams['font.family'] = ['DejaVu Sans']
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class EcommerceVisualizer:
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#16537e', '#8B5A3C', '#6A994E']
        
    def plot_missing_values(self, df, title="Eksik Değerler Analizi"):
        """Eksik değerleri görselleştir"""
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Eksik değer sayıları
        missing[missing > 0].plot(kind='bar', ax=ax1, color=self.colors[0])
        ax1.set_title('Eksik Değer Sayıları')
        ax1.set_ylabel('Eksik Değer Adedi')
        ax1.tick_params(axis='x', rotation=45)
        
        # Eksik değer yüzdeleri
        missing_percent[missing_percent > 0].plot(kind='bar', ax=ax2, color=self.colors[1])
        ax2.set_title('Eksik Değer Yüzdeleri')
        ax2.set_ylabel('Yüzde (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
    def plot_data_types(self, df, title="Veri Tipleri Dağılımı"):
        """Veri tiplerini görselleştir"""
        dtype_counts = df.dtypes.value_counts()
        
        plt.figure(figsize=(10, 6))
        colors = self.colors[:len(dtype_counts)]
        plt.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('equal')
        plt.show()
        
    def plot_category_distribution(self, df, column='category_code', top_n=15, title="Kategori Dağılımı"):
        """Kategori dağılımını görselleştir"""
        top_categories = df[column].value_counts().head(top_n)
        
        plt.figure(figsize=(14, 8))
        bars = plt.barh(range(len(top_categories)), top_categories.values, 
                       color=self.colors[0], alpha=0.8)
        plt.yticks(range(len(top_categories)), top_categories.index)
        plt.xlabel('Frekans')
        plt.title(f'En Popüler {top_n} {title}', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Bar'ların üzerine değerleri yaz
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + max(top_categories.values) * 0.01, 
                    bar.get_y() + bar.get_height()/2, 
                    f'{int(bar.get_width()):,}', 
                    va='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
    def plot_brand_analysis(self, df, top_n=15):
        """Marka analizi görselleştirmesi"""
        # Eksik markaları temizle
        df_clean = df.dropna(subset=['brand'])
        brand_counts = df_clean['brand'].value_counts().head(top_n)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # En popüler markalar
        bars1 = ax1.barh(range(len(brand_counts)), brand_counts.values, 
                        color=self.colors[2], alpha=0.8)
        ax1.set_yticks(range(len(brand_counts)))
        ax1.set_yticklabels(brand_counts.index)
        ax1.set_xlabel('Ürün Sayısı')
        ax1.set_title(f'En Popüler {top_n} Marka', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()
        
        # Bar değerlerini ekle
        for i, bar in enumerate(bars1):
            ax1.text(bar.get_width() + max(brand_counts.values) * 0.01,
                    bar.get_y() + bar.get_height()/2,
                    f'{int(bar.get_width()):,}',
                    va='center', fontsize=9)
        
        # Marka çeşitliliği
        total_brands = df_clean['brand'].nunique()
        top_brands_share = brand_counts.sum() / len(df_clean) * 100
        other_share = 100 - top_brands_share
        
        sizes = [top_brands_share, other_share]
        labels = [f'Top {top_n} Marka', 'Diğer Markalar']
        colors = [self.colors[2], self.colors[3]]
        
        ax2.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title(f'Marka Dağılımı\n(Toplam {total_brands:,} Marka)', 
                     fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
    def plot_price_analysis(self, df):
        """Fiyat analizi görselleştirmesi"""
        # Eksik fiyatları temizle
        df_clean = df.dropna(subset=['price'])
        prices = df_clean['price']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Fiyat dağılımı histogram
        ax1.hist(prices, bins=50, color=self.colors[0], alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Fiyat')
        ax1.set_ylabel('Frekans')
        ax1.set_title('Fiyat Dağılımı', fontweight='bold')
        ax1.axvline(prices.mean(), color='red', linestyle='--', label=f'Ortalama: ${prices.mean():.2f}')
        ax1.axvline(prices.median(), color='green', linestyle='--', label=f'Medyan: ${prices.median():.2f}')
        ax1.legend()
        
        # Log scale fiyat dağılımı
        ax2.hist(np.log1p(prices), bins=50, color=self.colors[1], alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Log(Fiyat + 1)')
        ax2.set_ylabel('Frekans')
        ax2.set_title('Fiyat Dağılımı (Log Scale)', fontweight='bold')
        
        # Box plot
        ax3.boxplot(prices, vert=True, patch_artist=True, 
                   boxprops=dict(facecolor=self.colors[2], alpha=0.7))
        ax3.set_ylabel('Fiyat')
        ax3.set_title('Fiyat Box Plot', fontweight='bold')
        ax3.set_xticklabels(['Fiyat'])
        
        # Fiyat kategorileri
        price_bins = [0, 50, 100, 250, 500, 1000, float('inf')]
        price_labels = ['0-50$', '50-100$', '100-250$', '250-500$', '500-1000$', '1000$+']
        df_clean['price_category'] = pd.cut(df_clean['price'], bins=price_bins, labels=price_labels)
        price_cat_counts = df_clean['price_category'].value_counts()
        
        bars4 = ax4.bar(range(len(price_cat_counts)), price_cat_counts.values, 
                       color=self.colors[3], alpha=0.8)
        ax4.set_xticks(range(len(price_cat_counts)))
        ax4.set_xticklabels(price_cat_counts.index, rotation=45)
        ax4.set_ylabel('Ürün Sayısı')
        ax4.set_title('Fiyat Kategorileri', fontweight='bold')
        
        # Bar değerlerini ekle
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
    def plot_category_price_relation(self, df, top_n=10):
        """Kategori-fiyat ilişkisi"""
        df_clean = df.dropna(subset=['category_code', 'price'])
        top_categories = df_clean['category_code'].value_counts().head(top_n).index
        df_top = df_clean[df_clean['category_code'].isin(top_categories)]
        
        plt.figure(figsize=(14, 8))
        
        # Box plot for top categories
        box_data = [df_top[df_top['category_code'] == cat]['price'].values 
                   for cat in top_categories]
        
        bp = plt.boxplot(box_data, labels=top_categories, patch_artist=True)
        
        # Renklendirme
        colors = self.colors * (len(bp['boxes']) // len(self.colors) + 1)
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.xlabel('Kategori')
        plt.ylabel('Fiyat')
        plt.title(f'Top {top_n} Kategoride Fiyat Dağılımı', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
    def plot_brand_price_relation(self, df, top_n=10):
        """Marka-fiyat ilişkisi"""
        df_clean = df.dropna(subset=['brand', 'price'])
        
        # En çok ürünü olan markaları al
        top_brands = df_clean['brand'].value_counts().head(top_n).index
        df_top = df_clean[df_clean['brand'].isin(top_brands)]
        
        # Ortalama fiyatları hesapla ve sırala
        brand_avg_prices = df_top.groupby('brand')['price'].agg(['mean', 'count']).round(2)
        brand_avg_prices = brand_avg_prices.sort_values('mean', ascending=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Ortalama fiyatlar
        bars1 = ax1.barh(range(len(brand_avg_prices)), brand_avg_prices['mean'], 
                        color=self.colors[4], alpha=0.8)
        ax1.set_yticks(range(len(brand_avg_prices)))
        ax1.set_yticklabels(brand_avg_prices.index)
        ax1.set_xlabel('Ortalama Fiyat ($)')
        ax1.set_title(f'Top {top_n} Markanın Ortalama Fiyatları', fontweight='bold')
        
        # Bar değerlerini ekle
        for i, bar in enumerate(bars1):
            ax1.text(bar.get_width() + max(brand_avg_prices['mean']) * 0.01,
                    bar.get_y() + bar.get_height()/2,
                    f'${bar.get_width():.0f}',
                    va='center', fontsize=10)
        
        # Box plot
        box_data = [df_top[df_top['brand'] == brand]['price'].values 
                   for brand in brand_avg_prices.index]
        
        bp = ax2.boxplot(box_data, labels=brand_avg_prices.index, patch_artist=True)
        
        # Renklendirme
        colors = self.colors * (len(bp['boxes']) // len(self.colors) + 1)
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xlabel('Marka')
        ax2.set_ylabel('Fiyat')
        ax2.set_title(f'Top {top_n} Markanın Fiyat Dağılımı', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
    def plot_time_analysis(self, df, datetime_col='event_time'):
        """Zaman analizi görselleştirmesi"""
        df_time = df.copy()
        df_time[datetime_col] = pd.to_datetime(df_time[datetime_col])
        
        # Tarih bileşenlerini çıkar
        df_time['date'] = df_time[datetime_col].dt.date
        df_time['hour'] = df_time[datetime_col].dt.hour
        df_time['day_of_week'] = df_time[datetime_col].dt.day_name()
        df_time['month'] = df_time[datetime_col].dt.month_name()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Günlük trend
        daily_counts = df_time['date'].value_counts().sort_index()
        ax1.plot(daily_counts.index, daily_counts.values, 
                color=self.colors[0], linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Tarih')
        ax1.set_ylabel('İşlem Sayısı')
        ax1.set_title('Günlük İşlem Trendi', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Saatlik dağılım
        hourly_counts = df_time['hour'].value_counts().sort_index()
        bars2 = ax2.bar(hourly_counts.index, hourly_counts.values, 
                       color=self.colors[1], alpha=0.8)
        ax2.set_xlabel('Saat')
        ax2.set_ylabel('İşlem Sayısı')
        ax2.set_title('Saatlik İşlem Dağılımı', fontweight='bold')
        ax2.set_xticks(range(0, 24, 2))
        
        # Haftalık dağılım
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_counts = df_time['day_of_week'].value_counts().reindex(day_order)
        bars3 = ax3.bar(range(len(weekly_counts)), weekly_counts.values, 
                       color=self.colors[2], alpha=0.8)
        ax3.set_xticks(range(len(weekly_counts)))
        ax3.set_xticklabels([day[:3] for day in day_order], rotation=45)
        ax3.set_ylabel('İşlem Sayısı')
        ax3.set_title('Haftalık İşlem Dağılımı', fontweight='bold')
        
        # Bar değerlerini ekle
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=9)
        
        # Aylık dağılım
        monthly_counts = df_time['month'].value_counts()
        ax4.pie(monthly_counts.values, labels=monthly_counts.index, autopct='%1.1f%%',
               colors=self.colors[:len(monthly_counts)], startangle=90)
        ax4.set_title('Aylık İşlem Dağılımı', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
    def plot_event_type_analysis(self, df, event_col='event_type'):
        """Event türü analizi"""
        if event_col not in df.columns:
            print(f"'{event_col}' sütunu bulunamadı!")
            return
            
        event_counts = df[event_col].value_counts()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart
        bars1 = ax1.bar(range(len(event_counts)), event_counts.values, 
                       color=self.colors[:len(event_counts)], alpha=0.8)
        ax1.set_xticks(range(len(event_counts)))
        ax1.set_xticklabels(event_counts.index, rotation=45)
        ax1.set_ylabel('Frekans')
        ax1.set_title('Event Türü Dağılımı', fontweight='bold')
        
        # Bar değerlerini ekle
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom')
        
        # Pie chart
        ax2.pie(event_counts.values, labels=event_counts.index, autopct='%1.1f%%',
               colors=self.colors[:len(event_counts)], startangle=90)
        ax2.set_title('Event Türü Oranları', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
    def create_summary_stats_table(self, df):
        """Özet istatistikler tablosu"""
        print("=" * 60)
        print("📊 VERİ SETİ ÖZET İSTATİSTİKLERİ")
        print("=" * 60)
        
        # Genel bilgiler
        print(f"📏 Boyut: {df.shape[0]:,} satır × {df.shape[1]} sütun")
        print(f"💾 Bellek kullanımı: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"📅 Analiz tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n" + "-" * 40)
        print("🔍 SÜTUN BİLGİLERİ")
        print("-" * 40)
        
        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            null_percent = (null_count / len(df)) * 100
            unique_count = df[col].nunique()
            
            print(f"• {col}")
            print(f"  - Tip: {dtype}")
            print(f"  - Eksik: {null_count:,} ({null_percent:.1f}%)")
            print(f"  - Benzersiz: {unique_count:,}")
            
            if dtype in ['int64', 'float64']:
                print(f"  - Min/Max: {df[col].min():.2f} / {df[col].max():.2f}")
                print(f"  - Ortalama: {df[col].mean():.2f}")
            print()