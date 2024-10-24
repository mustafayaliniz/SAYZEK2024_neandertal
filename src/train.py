
from ultralytics import YOLO  # Ultralytics YOLO kütüphanesini import ediyoruz

if __name__ == '__main__':
    # YOLO modelini 'models/yolo11l.pt' dosyasından yüklüyoruz
    model = YOLO('models/yolo11l.pt')

    # Modeli eğitiyoruz. Aşağıda eğitimin parametreleri yer alıyor:
    # data: Eğitim veri setinin bulunduğu yaml dosyasının yolu
    # epochs: Modeli 200 epoch boyunca eğiteceğiz
    # imgsz: Girdi boyutunu 512x512 olarak ayarlıyoruz
    # batch: Her batch'te 16 örnek kullanıyoruz
    # optimizer: AdamW optimizasyon algoritmasını kullanıyoruz
    # device: CUDA kullanarak eğitim işlemlerini GPU üzerinde yapacağız
    results = model.train(data="config/data.yaml", epochs=200, imgsz=512, batch=16, optimizer='AdamW', device="cuda")
