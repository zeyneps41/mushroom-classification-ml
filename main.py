
import pandas as pd
from pathlib import Path

# 1. Dosya yolu ve sütun isimleri
dosya_yolu = Path("data/agaricus-lepiota.data")

sutunlar = [
    "class", "cap-shape", "cap-surface", "cap-color",
    "bruises", "odor", "gill-attachment", "gill-spacing",
    "gill-size", "gill-color", "stalk-shape", "stalk-root",
    "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring",
    "veil-type", "veil-color", "ring-number", "ring-type",
    "spore-print-color", "population", "habitat"
]

# 2. Veriyi oku
df = pd.read_csv(
    dosya_yolu,
    header=None,
    names=sutunlar,
    dtype=str
)

print("ORIGINAL VERI BOYUTU:", df.shape)

# 3. Tekrar eden kayıtları kontrol et; silme yapma
print("\nTekrar eden satir sayisi:", df.duplicated().sum())

# 4. '?' işaretlerini gerçek eksik değerlere dönüştür
df = df.replace("?", pd.NA)

print("\n--- EKSİK DEĞER SAYILARI ---")
print(df.isna().sum().sort_values(ascending=False).head(10))

# 5. Kategorik sütunların değerlerini incele
print("\n--- KATEGORİ SAYILARI ---")
for sutun in df.columns:
    print(f"{sutun}: {df[sutun].nunique(dropna=True)} farkli kategori")

# 6. Hedef sütunundaki eksik değerleri kontrol et
print("\nHedef sütununda eksik değer:", df["class"].isna().sum())

# 7. Eksik stalk-root değerlerini 'unknown' ile doldur
# Satır silinmez; eksik olduğu bilgisi korunur.
df["stalk-root"] = df["stalk-root"].fillna("unknown")

# 8. Diğer sütunlarda eksik değer varsa raporla
print("\n--- DOLDURMA SONRASI EKSİK DEĞERLER ---")
print(df.isna().sum().sort_values(ascending=False).head(10))

# 9. Hedef sınıfları kontrol et
print("\n--- SINIF DAĞILIMI ---")
print(df["class"].value_counts())

# 10. Düzeltilmiş veri setini kaydet
Path("data").mkdir(exist_ok=True)

cikti_yolu = Path("data/mushroom_cleaned.csv")
df.to_csv(cikti_yolu, index=False)

print("\nTemizlenmis veri kaydedildi:", cikti_yolu)
print("Son veri boyutu:", df.shape)
print("Toplam satir kaybi:", 8124 - len(df))



from sklearn.model_selection import train_test_split

# 11. Ozellikleri (X) ve hedefi (y) ayir
X = df.drop(columns=["class"])
y = df["class"]

print("\n--- VERI AYIRMA ---")
print("Toplam veri:", len(df))
print("Ozellik sayisi:", X.shape[1])

# 12. Veriyi %80 egitim, %20 test olarak ayir
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 13. Kume boyutlarini goster
print("\nEgitim kayit sayisi:", len(X_train))
print("Test kayit sayisi:", len(X_test))

print("\nEgitim sinif dagilimi:")
print(y_train.value_counts())
print((y_train.value_counts(normalize=True) * 100).round(2))

print("\nTest sinif dagilimi:")
print(y_test.value_counts())
print((y_test.value_counts(normalize=True) * 100).round(2))

# 14. Ayrilan verileri kaydet
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("\nEgitim ve test dosyalari data klasorune kaydedildi.")


# 15. Makine ogrenmesi icin gerekli kutuphaneler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

# 16. Kategorik ozellikleri sayisal hale getir
on_isleme = ColumnTransformer(
    transformers=[
        (
            "kategorik",
            OneHotEncoder(handle_unknown="ignore"),
            list(X.columns)
        )
    ]
)

# 17. Kullanilacak algoritmalari tanimla
algoritmalar = {
    "SVM": SVC(),
    "k-NN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "Naive Bayes": BernoulliNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    )
}

# 18. Bes katli tabakali capraz dogrulama
capraz_dogrulama = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\n--- 5 KATLI CAPRAZ DOGRULAMA ---")

for isim, algoritma in algoritmalar.items():
    model = Pipeline([
        ("on_isleme", on_isleme),
        ("siniflandirici", algoritma)
    ])

    skorlar = cross_val_score(
        model,
        X_train,
        y_train,
        cv=capraz_dogrulama,
        scoring="accuracy",
        n_jobs=-1
    )

    print(
        f"{isim}: "
        f"Kat skorları = {skorlar.round(4)}, "
        f"Ortalama doğruluk = {skorlar.mean():.4f}"
    )


    

# 19. Test verisi uzerinde nihai degerlendirme
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import numpy as np

# Sonuclarin kaydedilecegi klasor
Path("results").mkdir(exist_ok=True)

sonuclar = []
roc_bilgileri = []

# p = pozitif sinif (zehirli mantar)
# e = negatif sinif (yenilebilir mantar)
y_test_binary = (y_test == "p").astype(int)

for isim, algoritma in algoritmalar.items():

    # On isleme ve algoritmayi birlestir
    model = Pipeline([
        ("on_isleme", on_isleme),
        ("siniflandirici", algoritma)
    ])

    # Modeli yalnizca egitim verisiyle egit
    model.fit(X_train, y_train)

    # Test verisinde tahmin yap
    y_pred = model.predict(X_test)

    # Karisiklik matrisi: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["e", "p"]
    )

    tn, fp, fn, tp = cm.ravel()

    # Temel metrikler
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, pos_label="p", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, pos_label="p", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, pos_label="p", zero_division=0
    )

    # Ek metrikler
    fpr = fp / (fp + tn) if (fp + tn) else 0
    fdr = fp / (fp + tp) if (fp + tp) else 0
    fnr = fn / (fn + tp) if (fn + tp) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0

    # Zehirli sinif icin olasilik/skor al
    siniflandirici = model.named_steps["siniflandirici"]

    if hasattr(model, "predict_proba"):
        olasiliklar = model.predict_proba(X_test)
        p_index = list(model.classes_).index("p")
        y_score = olasiliklar[:, p_index]
    else:
        # Bu modellerde karar skoru kullanilabilir
        y_score = model.decision_function(X_test)

    auc = roc_auc_score(y_test_binary, y_score)
    fpr_curve, tpr_curve, _ = roc_curve(y_test_binary, y_score)

    roc_bilgileri.append((isim, fpr_curve, tpr_curve, auc))

    # Sonuclari listeye ekle
    sonuclar.append({
        "Algoritma": isim,
        "Accuracy": accuracy,
        "FPR": fpr,
        "FDR": fdr,
        "FNR": fnr,
        "F1": f1,
        "Sensitivity": recall,
        "Specificity": specificity,
        "Precision": precision,
        "Recall": recall,
        "AUC": auc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

    # Karisiklik matrisi grafigini kaydet
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Edible (e)", "Poisonous (p)"]
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{isim} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"results/{isim.replace(' ', '_')}_confusion_matrix.png")
    plt.close()

# Sonuc tablosunu olustur
sonuc_df = pd.DataFrame(sonuclar)
sonuc_df.to_csv("results/model_metrics.csv", index=False)

print("\n--- TEST VERISI SONUCLARI ---")
print(sonuc_df.round(4).to_string(index=False))

# Butun algoritmalarin ROC egrilerini tek grafikte ciz
plt.figure(figsize=(9, 7))

for isim, fpr_curve, tpr_curve, auc in roc_bilgileri:
    plt.plot(
        fpr_curve,
        tpr_curve,
        label=f"{isim} (AUC = {auc:.4f})"
    )

plt.plot([0, 1], [0, 1], linestyle="--", label="Rastgele tahmin")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Eğrileri - Beş Algoritma")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/roc_curves.png")
plt.close()

print("\nSonuclar results/model_metrics.csv dosyasina kaydedildi.")
print("ROC grafigi results/roc_curves.png dosyasina kaydedildi.")
print("Karisiklik matrisi grafikleri results klasorune kaydedildi.")




# 20. OZELLIK SECIMI: Ki-kare (Chi-Square)
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report

print("\n--- KI-KARE ILE OZELLIK SECIMI ---")

# One-Hot Encoding'i yalnizca egitim verisine uydur
X_train_encoded = on_isleme.fit_transform(X_train, y_train)

# Kodlanmis ozellik adlarini al
ozellik_adlari = on_isleme.get_feature_names_out()

# Her kodlanmis ozelligin sinifla iliskisini puanla
ki_kare_skorlari, p_degerleri = chi2(
    X_train_encoded,
    y_train
)

ozellik_skorlari = pd.DataFrame({
    "Ozellik": ozellik_adlari,
    "Ki_kare_skoru": ki_kare_skorlari,
    "p_degeri": p_degerleri
})

ozellik_skorlari = ozellik_skorlari.sort_values(
    by="Ki_kare_skoru",
    ascending=False
)

# Puanlari dosyaya kaydet
ozellik_skorlari.to_csv(
    "results/feature_selection_scores.csv",
    index=False
)

print("\nEn yuksek puanli 15 kodlanmis ozellik:")
print(ozellik_skorlari.head(15).to_string(index=False))

# En iyi 10 kodlanmis ozelligi sec
secici = SelectKBest(score_func=chi2, k=10)
X_train_secili = secici.fit_transform(X_train_encoded, y_train)

# Secilen ozelliklerin isimleri
secilen_ozellikler = ozellik_adlari[secici.get_support()]

print("\nSecilen 10 ozellik:")
for ozellik in secilen_ozellikler:
    print("-", ozellik)

# Ayni on isleme ve secim adimlarini test verisine uygula
X_test_encoded = on_isleme.transform(X_test)
X_test_secili = secici.transform(X_test_encoded)

# Secilen ozelliklerle Random Forest egit
rf_secili = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_secili.fit(X_train_secili, y_train)
y_pred_secili = rf_secili.predict(X_test_secili)

# Baslangictaki Random Forest ve secilmis ozellikli modeli karsilastir
rf_normal = Pipeline([
    ("on_isleme", on_isleme),
    ("siniflandirici", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

rf_normal.fit(X_train, y_train)
y_pred_normal = rf_normal.predict(X_test)

karsilastirma = pd.DataFrame([
    {
        "Model": "Random Forest - Tum kodlanmis ozellikler",
        "Accuracy": accuracy_score(y_test, y_pred_normal),
        "F1": f1_score(
            y_test, y_pred_normal,
            pos_label="p",
            zero_division=0
        ),
        "Ozellik_sayisi": X_train_encoded.shape[1]
    },
    {
        "Model": "Random Forest - Secilen 10 ozellik",
        "Accuracy": accuracy_score(y_test, y_pred_secili),
        "F1": f1_score(
            y_test, y_pred_secili,
            pos_label="p",
            zero_division=0
        ),
        "Ozellik_sayisi": X_train_secili.shape[1]
    }
])

karsilastirma.to_csv(
    "results/feature_selection_comparison.csv",
    index=False
)

print("\n--- OZELLIK SECIMI KARSILASTIRMASI ---")
print(karsilastirma.to_string(index=False))

print("\nSecilmis modelin siniflandirma raporu:")
print(classification_report(
    y_test,
    y_pred_secili,
    labels=["e", "p"],
    target_names=["Yenilebilir (e)", "Zehirli (p)"],
    zero_division=0
))

print("\nOzellik secimi sonuclari results klasorune kaydedildi.")
