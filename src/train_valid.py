import os
import shutil
import random

# Klasör yolları
image_folder = 'utils/images'  # Görüntülerin bulunduğu ana klasör
label_folder = 'utils/labels'  # Etiketlerin bulunduğu ana klasör
train_image_folder = 'config/train/images'  # Eğitim için görüntülerin kaydedileceği klasör
train_label_folder = 'config/train/labels'  # Eğitim için etiketlerin kaydedileceği klasör
valid_image_folder = 'config/valid/images'  # Validasyon için görüntülerin kaydedileceği klasör
valid_label_folder = 'config/valid/labels'  # Validasyon için etiketlerin kaydedileceği klasör

# Eğitim için kullanılacak veri oranı (örn. %80 eğitim, %20 validasyon)
train_ratio = 0.8

# 1. Görüntü dosyalarını al (.png uzantılı)
image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

# 2. Dosyaları rastgele sırala (overfitting'i azaltmak için önemlidir)
random.shuffle(image_files)

# 3. Eğitim ve validasyon dosya sayılarını hesapla
train_size = int(len(image_files) * train_ratio)  # Eğitim setine düşen dosya sayısı

# 4. Eğitim ve validasyon dosyalarını ayır
train_files = image_files[:train_size]  # Eğitim seti dosyaları
valid_files = image_files[train_size:]  # Validasyon seti dosyaları


def copy_files(file_list, src_img_folder, src_lbl_folder, dst_img_folder, dst_lbl_folder):

    for file_name in file_list:
        # 5. Kaynak görüntü ve etiket dosyalarının tam yolunu al
        img_src = os.path.join(src_img_folder, file_name)  # Görüntü dosyası yolu
        lbl_src = os.path.join(src_lbl_folder, file_name.replace('.png', '.txt'))  # Etiket dosyası yolu

        # 6. Hedef görüntü ve etiket dosyalarının tam yolunu belirle
        img_dst = os.path.join(dst_img_folder, file_name)  # Hedef görüntü dosyası yolu
        lbl_dst = os.path.join(dst_lbl_folder, file_name.replace('.png', '.txt'))  # Hedef etiket dosyası yolu

        # 7. Dosyaları ilgili klasörlere kopyala
        shutil.copy(img_src, img_dst)  # Görüntü dosyasını kopyala
        shutil.copy(lbl_src, lbl_dst)  # Etiket dosyasını kopyala


# 8. Eğitim ve validasyon klasörleri yoksa oluştur (varsa hata vermez)
os.makedirs(train_image_folder, exist_ok=True)
os.makedirs(train_label_folder, exist_ok=True)
os.makedirs(valid_image_folder, exist_ok=True)
os.makedirs(valid_label_folder, exist_ok=True)

# 9. Eğitim seti dosyalarını kopyala
copy_files(train_files, image_folder, label_folder, train_image_folder, train_label_folder)

# 10. Validasyon seti dosyalarını kopyala
copy_files(valid_files, image_folder, label_folder, valid_image_folder, valid_label_folder)

# 11. İşlem tamamlandı mesajı
print(f"Eğitim için {len(train_files)} dosya, validasyon için {len(valid_files)} dosya kopyalandı.")
