# Büyük Veri Analizi ve Sınıflandırma Projesi

## 1. Projenin Amacı

Bu projenin amacı, mantarların özelliklerini kullanarak yenilebilir veya zehirli olduklarını sınıflandırmak ve farklı makine öğrenmesi algoritmalarının performanslarını karşılaştırmaktır.

## 2. Kullanılan Veri Seti

Projede UCI Mushroom veri seti kullanılmıştır.

- Toplam kayıt sayısı: 8124
- Özellik sayısı: 22
- Hedef değişken: `class`
- `e`: Yenilebilir mantar
- `p`: Zehirli mantar

Veri setinde eksik değerleri temsil eden `?` işaretleri tespit edilmiştir. `stalk-root` sütunundaki bu değerler `unknown` olarak değiştirilmiş, hiçbir satır silinmemiştir.

## 3. Veri Ön İşleme

- Veri seti incelenmiştir.
- Eksik değerler kontrol edilmiştir.
- Eksik değerler uygun bir kategoriyle doldurulmuştur.
- Kategorik değişkenler One-Hot Encoding yöntemiyle sayısal forma dönüştürülmüştür.
- Veri seti, sınıf oranları korunarak %80 eğitim ve %20 test verisi şeklinde ayrılmıştır.
- Model değerlendirmesinde 5 katlı Stratified Cross-Validation kullanılmıştır.

## 4. Kullanılan Sınıflandırma Algoritmaları

1. Support Vector Machine (SVM)
2. k-Nearest Neighbors (k-NN)
3. Random Forest
4. Naive Bayes
5. Logistic Regression

## 5. Değerlendirme Ölçütleri

Modeller aşağıdaki ölçütlerle karşılaştırılmıştır:

- Accuracy (Doğruluk)
- FPR (Yanlış Pozitif Oranı)
- FDR (Yanlış Keşif Oranı)
- FNR (Yanlış Negatif Oranı)
- F1-Score
- Sensitivity (Duyarlılık)
- Specificity (Özgüllük)
- Precision (Kesinlik)
- Recall
- ROC eğrisi ve AUC değeri

## 6. Özellik Seçimi

Ki-kare (Chi-Square) yöntemiyle özellik seçimi yapılmıştır. One-Hot Encoding sonrasında elde edilen 117 özellik arasından en yüksek puanlı 10 özellik seçilmiştir.

Random Forest sonuçları karşılaştırıldığında tüm özelliklerle doğruluk %100, seçilen 10 özellikle ise yaklaşık %97,78 olarak bulunmuştur. Böylece özellik sayısı azaltılmış, ancak küçük bir performans kaybı gözlenmiştir.

## 7. Sonuçlar

SVM, k-NN ve Random Forest modelleri test verisinde %100 doğruluk elde etmiştir. Logistic Regression yaklaşık %99,94 doğruluk göstermiştir. Naive Bayes ise yaklaşık %93,29 doğruluk elde etmiştir.

Sonuçlar, bu veri setinde kullanılan modellerin çoğunun yüksek sınıflandırma performansına sahip olduğunu göstermektedir. Bununla birlikte sonuçlar yalnızca kullanılan veri seti ve deney koşulları için geçerlidir.

## 8. Projeyi Çalıştırma

Gerekli kütüphaneleri yüklemek için:

```bash
python -m pip install -r requirements.txt
```

Projeyi çalıştırmak için:

```bash
python main.py
```

Sonuç dosyaları ve grafikler `results` klasörüne kaydedilmektedir.

## 9. Klasör Yapısı

```text
BuyukVeriOdevi/
├── data/
│   ├── agaricus-lepiota.data
│   ├── mushroom_cleaned.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
├── results/
│   ├── model_metrics.csv
│   ├── feature_selection_scores.csv
│   ├── feature_selection_comparison.csv
│   ├── roc_curves.png
│   └── confusion matrix grafikleri
├── main.py
├── requirements.txt
└── README.md
```

## 10. Güvenlik Notu

Bu proje eğitim ve makine öğrenmesi denemeleri amacıyla hazırlanmıştır. Gerçek hayatta mantarların yenilebilir olup olmadığına yalnızca bu tür bir modelin tahminine dayanarak karar verilmemelidir.