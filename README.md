# SAYZEK2024 Neandertal Projesi

Bu proje, Neandertal temalı nesne algılama sisteminin geliştirilmesi amacıyla tasarlanmıştır. Proje, YOLO11 tabanlı modelin eğitimi, çıkarımı ve değerlendirilmesi için gerekli olan tüm bileşenleri içermektedir. Aşağıda, projenin kurulumu, gereksinimleri, çalışma akışı ve diğer önemli bilgiler yer almaktadır.

## Proje Yapısı

```

SAYZEK2024_neandertal/
│
├── src/
│   ├── train.py                # Modelin eğitimini gerçekleştiren dosya.
│   ├── inference.py            # Eğitim sonrası tahminler yapar.
│   ├── detection.py            # Algılanan nesnelerin görselleştirilmesi.
│   ├── train_valid.py          # Eğitim ve validasyon süreçlerini yöneten dosya.
│   ├── models/                 # Eğitilen model dosyaları.
│   │   ├── best.pt             # En iyi model ağırlıkları.
│   ├── utils/                  # Yardımcı işlevleri içeren dosyalar.
│   │   ├── annotations/         # Annotations klasörü.
│   │   │   ├── train.json       # Eğitim için kullanılan annotation dosyası.
│   │   ├── images/             # Görüntü dosyaları. İndirilmesi Gerekiyor.
│   │   ├── data_loader.py      # Verilerin yüklenmesini sağlar.
│   │   ├── eval.py             # Modelin performansını değerlendirir.
│   │   ├── image_name_to_id.py # Görüntü isimlerini ID'ye çevirir.
│   └── config/                 # Konfigürasyon dosyaları.
│       ├── args.yaml           # Argüman yapılandırma dosyası.
│       ├── data.yaml           # Veri ayarları dosyası.
│
├── notebooks/
│   ├── model_development.ipynb  # Model geliştirme not defteri.
│   └── result.csv               # Model sonuçlarının saklandığı CSV dosyası.
│
├── requirements.txt             # Gerekli kütüphanelerin listesi.
│
└── README.md                    # Proje hakkında bilgi veren dosya.

```

## Kurulum Adımları

### Ortam Hazırlığı

1. **Python Versiyonu**: Bu proje Python 3.12 sürümü ile geliştirilmiştir. Python'un en güncel sürümünü [Python Resmi Sitesi](https://www.python.org/downloads/) üzerinden indirebilirsiniz.

2. **CUDA ve PyTorch**: CUDA 11.8 ve PyTorch 2.3.1+cu118 sürümleri kullanılmaktadır. CUDA, GPU hızlandırması için gereklidir ve PyTorch, derin öğrenme kütüphanesidir. Aşağıdaki komutları terminalde çalıştırarak gerekli kütüphaneleri kurabilirsiniz.

### Gerekli Kütüphanelerin Kurulumu

3. **Gerekli Kütüphanelerin Yüklenmesi**: Projeyi çalıştırmak için aşağıdaki adımları izleyin:

   - **Gerekli Kütüphanelerin Listesi**: `requirements.txt` dosyası, projede kullanılan tüm kütüphaneleri ve sürümlerini içerir. Bu dosyayı kullanarak kütüphaneleri yükleyebilirsiniz.

   ```bash
   pip install -r requirements.txt
   ```
   
## Modelin Eğitimi ve Çıkarımı

Projenin ana dosyaları belirli bir sırayla çalıştırılmalıdır. Her bir dosya, belirli bir işlemi gerçekleştirmek üzere tasarlanmıştır.

1. **Veri Yükleme**: `data_loader.py` dosyası, gerekli veri setini yükler ve uygun formatta işlenmesini sağlar. Bu dosya, JSON dosyasını YOLO formatına dönüştürerek her bir resim için bir TXT dosyası oluşturur. Her TXT dosyası, o resme karşılık gelen etiket ve koordinat bilgilerini içerir. Veri setini yüklemek için bu dosyayı çalıştırın:

   ```bash
   python src/data_loader.py
   ```
   
2. **Eğitim ve Validasyon**: `train_valid.py` dosyası, modelin eğitim ve validasyon süreçlerini yönetir. Bu dosya, görüntüleri ve etiketleri %80 eğitim ve %20 validasyon oranında ayırır. Eğitim seti, modelin öğrenmesi için kullanılırken, validasyon seti modelin performansını değerlendirmek için kullanılır. Modelin eğitim sürecini başlatmak için bu dosyayı çalıştırın:

   ```bash
   python src/train_valid.py
   ```

3. **Model Eğitimi**: `train.py` dosyası, modelin eğitimini gerçekleştirir. Eğitim süreci sırasında modelin ağırlıkları güncellenir.

   ```bash
   python src/train.py
   ```

4. **Görüntü Tanıma**: `inference.py` dosyası, eğitilen model ile tahminler yapar. Bu adımda, modelin tahmin ettiği nesneler üzerinde çıkarım yapılır.

   ```bash
   python src/inference.py
   ```

5. **Görüntü İsimlerini ID'lere Dönüştürme**: `image_name_to_id.py` dosyası, `train.json` dosyasından gelen ID'leri işleyerek JSON formatında kaydeder. Bu adım, modelin değerlendirilmesi için gerekli olan ID eşleşmelerini sağlar.

   ```bash
   python src/image_name_to_id.py
   ```

6. **Sonuç Değerlendirmesi**: `eval.py` dosyası, modelin performansını değerlendirir ve başarı oranlarını hesaplar. Bu adımda, daha önce çalıştırılan `image_name_to_id.py` dosyasından elde edilen ID'ler kullanılır.

   ```bash
   python src/eval.py
   ```

7. **Sonuç Görselleştirme**: `detection.py` dosyası, algılanan nesneleri görselleştirir ve sonuçları kullanıcıya sunar.

   ```bash
   python src/detection.py
   ```

## Minimum Donanım Gereksinimleri

- **CPU**: En az 4 çekirdekli bir işlemci.
- **RAM**: 16 GB RAM (daha fazla bellek önerilir).
- **GPU**: CUDA uyumlu bir GPU (örneğin, NVIDIA RTX 2060 veya daha iyisi) ile modelin eğitim süresi ve performansı önemli ölçüde iyileşir.
- **Disk Alanı**: Proje dosyaları ve veri setleri için en az 10 GB boş alan.

## Özel GPU Gereksinimleri

- Proje CUDA 11.8 ile uyumludur; bu nedenle, NVIDIA GPU’ların en güncel sürücülerinin kurulu olması gerekmektedir.
- CUDA ve cuDNN’in uyumlu sürümlerinin yüklenmesi önerilmektedir.

## Ek Notlar ve Öneriler

- Projenin düzgün çalışabilmesi için gerekli tüm dosyaların ve klasörlerin doğru şekilde yapılandırıldığından emin olun.
- Eğitim sürecinde, yeterli GPU belleğine sahip bir cihaz kullanmanız önerilir. Eğitim sırasında belleği aşmamaya dikkat edin.
- Hata ayıklama sırasında, her bir dosyanın işlevlerini ve hata mesajlarını dikkatlice inceleyin. Hataları gidermek için dökümantasyonu ve topluluk forumlarını kullanabilirsiniz.
- `data.yaml` dosyasında, `train:` ve `val:` anahtarlarının ardından gelen dosya uzantılarını kendi bilgisayarınıza göre güncellemeyi unutmayın.

## İletişim

Herhangi bir soru veya geri bildirim için lütfen [uzayk204@gmail.com](mailto:uzayk204@gmail.com) veya [mhmtture04@gmail.com](mailto:mhmtture04@gmail.com) adreslerinden bizimle iletişime geçebilirsiniz.

