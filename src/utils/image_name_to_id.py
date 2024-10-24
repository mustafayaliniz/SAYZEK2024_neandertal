import json
from pycocotools.coco import COCO

# COCO anotasyon dosyasını yükle
annotation_file_path = 'annotations/train.json'  # COCO formatındaki anotasyon dosyasının yolu
coco_ann = COCO(annotation_file=annotation_file_path)

# Görüntü dosya adlarını image_id ile eşleştir
imgfile2imgid = {coco_ann.imgs[i]['file_name']: i for i in coco_ann.imgs.keys()}

# Elde edilen eşleşmeleri JSON dosyasına kaydet
output_file_path = 'image_file_name_to_image_id_train.json'  # Eşleşmelerin kaydedileceği JSON dosyasının yolu
with open(output_file_path, 'w') as f:
    json.dump(imgfile2imgid, f, indent=4)  # Daha okunaklı bir format için indent eklenmiştir

print(f"Eşleşmeler başarıyla '{output_file_path}' dosyasına kaydedildi.")
