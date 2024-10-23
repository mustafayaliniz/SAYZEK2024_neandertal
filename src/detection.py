import cv2  # OpenCV kütüphanesini içe aktar
from ultralytics import YOLO  # Ultralytics YOLO modelini içe aktar

# Modeli yükle
model = YOLO("runs/detect/train3/weights/best.pt")  # Eğitilmiş YOLO modelinin ağırlık dosyasını yükle

# Görüntü dosyasının yolu
image_path = "utils/images/12513_104_30.png"  # İşlenecek görüntünün yolu

# Görüntüyü oku
img = cv2.imread(image_path)  # Belirtilen yoldan görüntüyü oku

# Sınıf adlarının listesi
classification = ['bina', 'yol_kesisimi', 'futbol_sahasi', 'silo']  # Algılanacak nesnelerin sınıf adları

# Görüntü okunamazsa hata mesajı yazdır
if img is None:
    print(f"Görüntü okunamadı: {image_path}")
else:
    # Nesne algılama
    results = model(img)  # Görüntü üzerinde nesne algılama işlemini gerçekleştir

    # Algılanan kutuları çiz ve doğruluk değerlerini yazdır
    for r in results:  # Algılanan sonuçlar üzerinde döngü başlat
        for box in r.boxes:  # Her bir algılanan kutu için döngü
            # Kutu koordinatlarını al
            x1, y1, x2, y2 = map(int, box.xyxy[0][:4])  # Kutunun sol üst ve sağ alt köşe koordinatlarını al
            confidence = box.conf[0].item()  # Algılanan nesnenin doğruluk değerini al
            class_id = int(box.cls[0].item())  # Algılanan nesnenin sınıf ID'sini al

            # Kutuyu çiz
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)  # Algılanan kutunun etrafına dikdörtgen çiz

            # Doğruluk değerini ve sınıf adını yan yana yaz
            box_label = f"{classification[class_id]} {confidence:.2f}"  # Sınıf adını ve doğruluk değerini birleştir
            cv2.putText(img, box_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Metni kutunun üzerine yaz

    # Sonuçları göster
    cv2.imshow('Neandertal Sayzek Datathon 2024', img)  # İşlenen görüntüyü göster

    # 'q' tuşuna basarak çıkış yap
    cv2.waitKey(0)  # Herhangi bir tuşa basılana kadar bekle

# Tüm pencereleri kapat
cv2.destroyAllWindows()  # Tüm OpenCV pencerelerini kapat
