# ⚡ TenderIQ — İhale Optimizasyon Motoru

> **Emlak Konut Ideathon 2025** — Akıllı Değerleme ve Yatırım Analitiği

---

## 🎯 Problem

Emlak Konut her yıl 40-50 ihale gerçekleştiriyor. Her ihalede en kritik karar şu:

> *"Bu arsayı yüzde kaç gelir paylaşımıyla ihaleye çıkaralım?"*

Bu karar şu an tecrübe ve sezgiyle alınıyor. Yanlış belirlenen her oran:

- Çok düşük → az müteahhit gelir, rekabet olmaz, kötü teklif alınır
- Çok yüksek → müteahhit kalitesi düşer, gecikme riski artar, gelir azalır
- Doğru oran → maksimum rekabet, maksimum gelir, güvenilir müteahhit

Yılda 40-50 ihalede %2-3'lük oran hatası = **yüz milyonlarca TL kayıp.**

---

## 💡 Çözüm

TenderIQ, geçmiş ihale verilerinden öğrenen bir yapay zeka modelidir. Her arsa parseli için:

- Optimal gelir paylaşımı oranını tahmin eder
- Hangi müteahhit profilinin teklif vereceğini öngörür
- Projenin kaç ayda tamamlanacağını hesaplar
- Farklı senaryoların gelir etkisini karşılaştırır

---

## 📊 Sistem Mimarisi

### Veri Katmanı

**İç Veri (Emlak Konut'un mevcut verisi):**

| Veri | Açıklama |
|------|----------|
| Geçmiş ihale oranları | Her ihaleye çıkılan oran ve sonucu |
| Teklif kayıtları | Kaç müteahhit teklif verdi, kimler verdi |
| Müteahhit performansı | O projeyi kaç günde, ne kalitede tamamladı |
| Hak ediş kayıtları | Ödeme tarihleri ve miktarları |
| Milestone verileri | Temel, kaba inşaat, ince işler, iskan tarihleri |

**Dış Veri (Açık kaynaklardan otomatik çekilen):**

| Veri | Kaynak | Güncelleme |
|------|--------|------------|
| İnşaat maliyet endeksi | TMB (Türkiye Müteahhitler Birliği) | Aylık |
| Çelik / çimento fiyatları | TCMB, TMB | Haftalık |
| Döviz kuru | TCMB API | Günlük |
| Konut satış fiyatları | TÜİK, REIDIN | Aylık |
| Müteahhit kapasitesi | SGK işçi kayıtları | Aylık |
| Bölgesel talep endeksi | TÜİK, REIDIN | Aylık |

---

### Feature Engineering (Model Girdileri)

Ham verilerden türetilen özellikler:

```
Lokasyon Skoru        = Şehir + Bölge tipi + Ulaşım erişimi + Emsal fiyat
Piyasa Sıcaklığı      = Son 6 ay satış hızı + Güncel stok/talep oranı
Müteahhit Arz Endeksi = Aktif ihaleler / Piyasadaki yetkin müteahhit sayısı
Maliyet Baskısı       = Döviz kuru + Çelik endeksi + İşçilik maliyeti
Proje Çekicilik Skoru = Alan + İmar hakkı + Tahmini satış geliri
Rekabet Yoğunluğu     = Benzer dönem ihalelerindeki ortalama teklif sayısı
```

---

### Model Katmanı

#### Model 1 — Teklif Sayısı Tahmini
**Soru:** Bu oran ve koşullarda kaç müteahhit teklif verir?

```
Algoritma : XGBoost Regressor
Girdi     : Oran + Lokasyon skoru + Piyasa sıcaklığı + Maliyet baskısı
Çıktı     : Tahmini teklif sayısı aralığı (min - max)
Eğitim    : Geçmiş 200+ Emlak Konut ihalesi
```

#### Model 2 — Gelir Optimizasyonu
**Soru:** Hangi oran Emlak Konut'a en yüksek geliri getirir?

```
Algoritma : LightGBM + Scipy Optimize
Girdi     : Tüm feature'lar + Model 1 çıktısı
Çıktı     : Optimal oran + Güven aralığı + Tahmini toplam gelir
Metrik    : Beklenen Gelir = P(başarılı teklif) × Tahmini proje geliri
```

#### Model 3 — Süre Tahmini
**Soru:** Bu müteahhit profiliyle proje kaç ayda biter?

```
Algoritma : LightGBM + Müteahhit geçmiş performans vektörü
Girdi     : Müteahhit profili + Proje büyüklüğü + Piyasa koşulları
Çıktı     : Tahmini tamamlanma süresi (ay) + Gecikme riski skoru
```

---

### Output Örneği

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARSA        : Ankara Yenimahalle, 12.400 m²
TARİH       : Mart 2025
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPTİMAL ORAN : %33  (güven aralığı: %31–%35)

SENARYO KARŞILAŞTIRMASI:

  %30 → Teklif: 3-5   | Gelir: ₺1.9Mrd | Süre: 42 ay ⚠️
  %33 → Teklif: 8-12  | Gelir: ₺2.4Mrd | Süre: 35 ay ✅
  %36 → Teklif: 1-3   | Gelir: ₺1.8Mrd | Süre: 44 ay ❌

ETKİ FAKTÖRLERİ:
  Piyasa Talebi      %38  ████████░░
  Lokasyon Değeri    %30  ██████░░░░
  Döviz Baskısı      %22  ████░░░░░░
  Proje Büyüklüğü    %10  ██░░░░░░░░

AKSİYON ÖNERİLERİ:
  → %33 ile ihaleye çıkın
  → Orta-büyük ölçekli, Ankara deneyimli firmalar bekleniyor
  → Mevcut ortalama karara göre tahmini ek gelir: +₺312M
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📈 Metrikler ve Başarı Kriterleri

### Model Performans Metrikleri

| Metrik | Hedef | Açıklama |
|--------|-------|----------|
| Teklif Sayısı MAE | ≤ 2 | Tahmin edilen vs gerçek teklif sayısı farkı |
| Oran Tavsiyesi Doğruluğu | ≥ %75 | Önerilen oranın ±2 puan içinde kalması |
| Gelir Tahmini MAPE | ≤ %12 | Tahmini gelir hata payı |
| Süre Tahmini MAE | ≤ 45 gün | Tamamlanma süresi sapması |

### İş Etkisi Metrikleri

| Metrik | Mevcut Durum | TenderIQ ile |
|--------|-------------|--------------|
| İhale başına ortalama teklif sayısı | Bilinmiyor | Ölçülebilir ve optimize edilebilir |
| Oran belirleme süresi | 1-2 hafta | 1 gün |
| Veri tabanlı karar oranı | %0 | %100 |
| Tahmini ek gelir (yıllık) | — | Mevcut portföyün %8-14'ü |

---

## 🛠️ Teknik Stack

```
Veri Toplama    : Python + Scheduled API calls (TCMB, TÜİK, TMB)
Veri İşleme     : Pandas, NumPy
Modelleme       : XGBoost, LightGBM, Scipy Optimize
Yorumlanabilirlik: SHAP (model kararlarını açıklar)
API             : FastAPI
Frontend        : React + Recharts
Deployment      : Docker
```

---

## 🚀 Pilot Plan

Büyük altyapı yatırımı gerekmiyor.

**Ay 1:** Emlak Konut'un geçmiş 50-100 ihale verisi ile model eğitimi

**Ay 2:** Seçilen 3 aktif ihale üzerinde gölge test (sistem tahmin eder, karar insan verir)

**Ay 3:** Gerçek sonuçlarla karşılaştırma, model iyileştirmesi

**Ay 4+:** Tüm ihale sürecine entegrasyon

---

## 👥 Takım
Şeyda Altın 
Layan Bassout 
Alaa Madı
Emlak Konut Ideathon 2026 — Ankara 

---

## 📎 Demo

🔗 [Canlı Demo](https://kullaniciadın.github.io/tenderiq)

> Demo, gerçek veriler olmaksızın kural tabanlı bir yaklaşımla çalışmaktadır.
> Gerçek model, Emlak Konut'un tarihsel ihale verileriyle eğitilecektir.

---

*TenderIQ — Emlak Konut'un milyarlarca TL'lik ihale kararını ilk kez veriyle almasını sağlar.*
