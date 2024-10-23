import json
import os

# Train JSON dosyasını yükleyin
with open("annotations/train.json", "r") as file:
    data = json.load(file)

# Etiketlerin kaydedileceği klasör
label_folder = 'labels'


# Etiketleri YOLO formatına dönüştürme fonksiyonu
def convert_to_yolo_format(annotation, image_width, image_height):

    x_min, y_min, box_width, box_height = annotation["bbox"]  # COCO formatında bbox bilgisi xywh şeklindedir
    x_max = x_min + box_width  # x_max değeri
    y_max = y_min + box_height  # y_max değeri

    # YOLO formatı için gerekli center_x, center_y, genişlik ve yükseklik hesaplaması
    center_x = (x_min + x_max) / 2 / image_width
    center_y = (y_min + y_max) / 2 / image_height
    width = box_width / image_width
    height = box_height / image_height

    # category_id'yi -1 yaparak 0 tabanlı sınıf id'si oluşturuyoruz (YOLO formatında bu şekilde gereklidir)
    return f'{annotation["category_id"] - 1} {center_x} {center_y} {width} {height}'


# Etiket dosyalarının kaydedileceği klasörü oluştur
os.makedirs(label_folder, exist_ok=True)

# Her bir görüntü için ilgili etiketleri txt dosyası olarak kaydetme
for image in data["images"]:
    image_id = image["id"]  # Görüntünün ID'sini al
    image_width = image["width"]  # Görüntünün genişliğini al
    image_height = image["height"]  # Görüntünün yüksekliğini al
    file_name = os.path.splitext(image["file_name"])[0]  # Dosya adını uzantısız şekilde al

    # Bu görüntüye ait etiketleri bul
    annotations = [ann for ann in data["annotations"] if ann["image_id"] == image_id]

    # Her bir görüntü için bir txt dosyası oluştur ve etiketleri yaz
    with open(os.path.join(label_folder, f"{file_name}.txt"), "w") as txt_file:
        for ann in annotations:
            # Etiketi YOLO formatına dönüştür
            yolo_format = convert_to_yolo_format(ann, image_width, image_height)
            # YOLO formatındaki etiketi dosyaya yaz
            txt_file.write(yolo_format + "\n")

# Tamamlandığında mesaj yazdır
print('TXT dosyaları labels klasöründe oluşturuldu.')
