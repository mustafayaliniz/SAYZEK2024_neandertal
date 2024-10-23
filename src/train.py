from ultralytics import YOLO

def train_model(model_path, data_config, epochs, img_size, device, batch_size, optimizer):

    # 1. YOLO modelini yükle
    model = YOLO(model_path)

    # 2. Modeli eğitim moduna geçir ve eğitimi başlat
    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=img_size,
        device=device,
        batch=batch_size,
        optimizer=optimizer
    )

    # 3. Eğitim sonuçlarını döndür
    return results

if __name__ == '__main__':
    # Model ve eğitim parametreleri
    model_path = 'models/yolo11l.pt'  # Eğitilecek modelin ağırlık dosyasının yolu
    data_config = 'config/data.yaml'  # Eğitim verisinin yapılandırma dosyası
    epochs = 200  # Epoch sayısı
    img_size = 512  # Görüntü boyutu
    device = 'cuda'  # GPU ile eğitim
    batch_size = 16  # Batch boyutu
    optimizer = 'AdamW'  # Optimizer türü

    # 4. Modeli eğit
    train_results = train_model(model_path, data_config, epochs, img_size, device, batch_size, optimizer)

    # 5. Eğitim sonuçlarını yazdır (isteğe bağlı)
    print(f"Eğitim tamamlandı. Sonuçlar: {train_results}")
