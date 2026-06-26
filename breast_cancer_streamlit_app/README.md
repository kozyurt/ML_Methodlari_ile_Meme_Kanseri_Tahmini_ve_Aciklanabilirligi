# 🔬 Meme Kanseri Tahmin Sistemi — Streamlit Uygulaması

## 📁 Klasör Yapısı

```
breast_cancer_app/
├── app.py                          ← Ana uygulama
├── breast_cancer_pipeline.pkl      ← Eğitilmiş model (notebook'tan kopyala)
├── model_info.pkl                  ← Model meta bilgileri
├── requirements.txt                ← Python bağımlılıkları
└── graphs/
    ├── model_comparison_eng.png
    ├── final_model_confusion_matrix.png
    ├── shap_importance_eng.png
    ├── shap_summary_eng.png
    ├── correlation_matrix_eng.png
    ├── diagnosis_distribution_eng.png
    ├── target_correlation_eng.png
    └── top10_outliers_boxplot_eng.png
```

## 🚀 Yerel Çalıştırma

```bash
# 1. Bağımlılıkları kur
pip install -r requirements.txt

# 2. Uygulamayı başlat
streamlit run app.py
```

## ☁️ Streamlit Community Cloud'a Deploy

1. GitHub'a push et:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/KULLANICI_ADI/REPO_ADI.git
   git push -u origin main
   ```

2. [share.streamlit.io](https://share.streamlit.io) adresine git
3. "New app" → GitHub reponuzu seç
4. Main file: `app.py` olarak ayarla
5. Deploy!

## ⚠️ ÖNEMLİ

`breast_cancer_pipeline.pkl` dosyasını notebook'tan bu klasöre kopyalamayı unutma!
Notebook'ta şu satır ile kaydedildi:
```python
pickle.dump(final_pipeline, open('breast_cancer_pipeline.pkl', 'wb'))
```

## 📊 Uygulama Özellikleri

- **🎯 Tahmin Yap**: 30 özelliği manuel girerek anlık tahmin
- **📊 Model Analizi**: 9 model karşılaştırması, confusion matrix, ROC eğrisi
- **🔍 Veri Keşfi**: Korelasyon matrisi, dağılım grafikleri, SHAP analizi
- **📖 Proje Hakkında**: Pipeline akışı ve metodoloji açıklaması
