import json
import cv2
import os
from ultralytics import YOLO  # YOLO11l için ultralytics kütüphanesini kullanıyoruz

def run_inference(model_path, test_images_path, json_output_path, image_id_mapping_path):

    # 1. Görüntü dosyası adı ile image_id eşleşmelerini JSON dosyasından yükle
    image_file_name_to_image_id = json.load(open(image_id_mapping_path))

    # 2. YOLO11l modelini yükle
    model = YOLO(model_path)

    results = []  # Tahmin sonuçlarını saklayacak liste

    # 3. Test görüntüleri üzerinde tahmin yap
    for img_name in os.listdir(test_images_path):
        image_path = os.path.join(test_images_path, img_name)
        image = cv2.imread(image_path)  # Görüntüyü yükle

        #  ----------------- Model Tahmini ve Sonuçların İşlenmesi -----------------  #
        # Model kullanılarak tahmin yap
        results_model = model(image)

        bboxes, labels, scores = [], [], []

        # Tahmin sonuçlarını parse et (her bir tahmin için bbox, label, score bilgilerini al)
        for pred in results_model[0].boxes:
            bbox = pred.xyxy[0].cpu().numpy()  # Bbox bilgileri xyxy formatında gelir
            bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])  # bbox bilgilerini kaydet
            labels.append(int(pred.cls))  # sınıf etiketini kaydet
            scores.append(pred.conf.cpu().numpy())  # confidence score kaydet
        #  ----------------- Model Tahmini ve Sonuçların İşlenmesi -----------------  #

        # 4. Görüntü adından image_id'yi al
        img_id = image_file_name_to_image_id[img_name]

        # 5. Her bbox, label ve score için JSON'a uygun formatta sonuç ekle
        for bbox, label, score in zip(bboxes, labels, scores):
            # xyxy formatındaki bbox'u xywh formatına çeviriyoruz
            bbox[2], bbox[3] = bbox[2] - bbox[0], bbox[3] - bbox[1]

            # Sonuçları JSON formatına uygun hale getir
            res = {
                'image_id': img_id,
                'category_id': int(label) + 1,  # Modelin sınıfı 0'dan başlıyorsa 1 ekleyin
                'bbox': list(map(float, bbox)),  # Bbox verilerini float tipine çevir
                'score': float("{:.8f}".format(float(score)))  # Score'u daha okunaklı hale getir
            }
            results.append(res)  # Her bir sonucu sonuçlar listesine ekle

    # 6. Sonuçları JSON dosyasına kaydet
    with open(json_output_path, 'w') as f:
        json.dump(results, f, indent=4)  # Daha okunaklı olması için indent kullanarak kaydediyoruz

    print(f"JSON dosyası '{json_output_path}' başarıyla oluşturuldu.")


if __name__ == '__main__':
    # Model, test görüntüleri ve JSON dosyası yolları
    model_path = "runs/detect/train3/weights/best.pt"  # Eğitilen modelin yolu
    test_images_path = 'config/train/images'  # Test için kullanılacak görüntülerin yolu
    json_output_path = 'utils/neandertal.json'  # Sonuçların kaydedileceği JSON dosyasının yolu
    image_id_mapping_path = 'utils/image_file_name_to_image_id_train.json'  # image_id eşleşmelerini içeren JSON dosyası

    # Tahmin işlemini başlat
    run_inference(model_path, test_images_path, json_output_path, image_id_mapping_path)
