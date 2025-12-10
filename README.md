# ML Methodları ile Meme Kanseri Tahmini ve Acıklanabilirliği

Bu proje, Wisconsin Meme Kanseri (Tanı) veri setini kullanarak meme kanseri malignite (kötü huylu) durumunu tahmin etmek için kapsamlı bir makine öğrenimi çözümü sunmaktadır. Proje, veri ön işleme, özellik mühendisliği, çeşitli makine öğrenimi modellerinin karşılaştırılması, en iyi modelin seçimi, hiperparametre optimizasyonu ve modelin açıklanabilirliğini içeren uçtan uca bir iş akışını kapsamaktadır.

## Proje Hedefleri

- Meme kanseri teşhisinde yüksek doğruluk ve güvenilirlik sağlayan bir sınıflandırma modeli geliştirmek.
- Modelin kararlarını anlamak ve yorumlamak için SHAP (SHapley Additive exPlanations) gibi açıklanabilirlik tekniklerini kullanmak.
- Modelin üretim ortamına kolayca entegre edilebilmesi için uçtan uca bir makine öğrenimi pipeline'ı oluşturmak.
- Klinik karar verme süreçlerine destek olabilecek, şeffaf ve güvenilir bir araç sunmak.

## Kullanılan Teknolojiler ve Kütüphaneler

- **Python**
- **Pandas** ve **NumPy** (Veri işleme ve manipülasyonu)
- **Scikit-learn** (Makine öğrenimi modelleri, pipeline, ön işleme araçları)
- **Matplotlib** ve **Seaborn** (Veri görselleştirme)
- **SHAP** (Model açıklanabilirliği)
- **Pickle** (Model kaydetme/yükleme)

## Proje Adımları

1.  **Veri Keşfi ve Ön İşleme:** Veri setinin anlaşılması, eksik değerlerin ve aykırı değerlerin incelenmesi.
2.  **Özellik Mühendisliği:** Mevcut özelliklerden yeni, daha açıklayıcı özellikler türetilmesi (örn. `concavity_points_product`).
3.  **Model Karşılaştırması (Baseline):** Logistic Regression, SVM, Random Forest, Gradient Boosting, XGBoost, LightGBM, KNN, Naive Bayes ve Decision Tree gibi çeşitli sınıflandırma modellerinin performanslarının karşılaştırılması.
4.  **En İyi Model Seçimi:** F1-Skoru, ROC-AUC ve genel dengeye göre en iyi performans gösteren modelin (Logistic Regression) belirlenmesi.
5.  **Hiperparametre Optimizasyonu:** Seçilen model için GridSearchCV ile hiperparametre ayarı yapılması.
6.  **Ensemble Model Denemesi:** Birden fazla güçlü modelin birleştirilmesiyle ensemble (oylama) modelinin oluşturulması ve performansının değerlendirilmesi.
7.  **Model Açıklanabilirliği (XAI):** SHAP değerleri kullanılarak modelin global ve lokal düzeyde nasıl kararlar verdiğinin analiz edilmesi. Özellik önem sıralaması, özellik etki analizi ve bireysel tahminler için waterfall grafikleri ile modelin şeffaflığı artırılmıştır.
8.  **Uçtan Uca Pipeline Oluşturma:** Özellik mühendisliği, ölçeklendirme ve nihai modeli içeren bir `sklearn.pipeline` nesnesinin oluşturulması.
9.  **Model Kaydetme ve Yükleme:** Eğitilmiş pipeline'ın ve model meta bilgilerinin kalıcı olarak kaydedilmesi ve tekrar yüklenerek doğrulanması.
10. **Yeni Veri ile Tahmin:** Kaydedilen pipeline'ın yeni, görülmemiş veriler üzerinde tahmin yapmak için kullanılması ve gerçek dünya senaryosunun simüle edilmesi.

## Kurulum ve Kullanım

1.  Depoyu klonlayın:
    ```bash
    git clone https://kozyurt/ML_Methodlari_ile_Meme_Kanseri_Tahmini_ve_Aciklanabilirligi.git
    cd ML_Methodlari_ile_Meme_Kanseri_Tahmini_ve_Aciklanabilirligi
    ```
2.  Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install -r requirements.txt
    ```
3.  Jupyter Notebook'u başlatın ve `breast_cancer_prediction.ipynb` dosyasını açın:
    ```bash
    jupyter notebook
    ```
4.  Notebook'taki adımları takip ederek projeyi çalıştırın ve sonuçları inceleyin.

## Katkıda Bulunma

Bu projeye katkıda bulunmaktan çekinmeyin! Her türlü geri bildirim, hata düzeltmesi veya yeni özellik önerisi memnuniyetle karşılanır.
