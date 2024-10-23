from pycocotools.coco import COCO  # COCO formatındaki anotasyonları işlemek için gerekli kütüphane
from pycocotools.cocoeval import COCOeval  # COCO formatında değerlendirme yapmak için kullanılan kütüphane
import argparse  # Komut satırından argümanları almak için kullanılan kütüphane


def eval_json(ann_file, det_file):

    # Anotasyon dosyasını yükle
    coco_gt = COCO(ann_file)

    # Tespit sonuçlarını anotasyon dosyasına yükle
    coco_dt = coco_gt.loadRes(det_file)

    # COCO değerlendirme nesnesi oluştur
    cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')

    # Değerlendirme işlemini gerçekleştir
    cocoEval.evaluate()
    cocoEval.accumulate()  # Değerlendirme sonuçlarını toplar
    cocoEval.summarize()  # Değerlendirme özetini gösterir

    # mAP@50 metriğini al
    map50 = cocoEval.stats[1]

    # mAP@50'yi beş ondalık basamağa yuvarla
    map50 = round(float(map50), 5)

    return map50


def main():

    # Argümanları tanımla
    parser = argparse.ArgumentParser(
        description="COCO API kullanarak bir tespit dosyasını anotasyon dosyasına göre değerlendirir.")

    # Anotasyon dosyası için argüman
    parser.add_argument('--ann_file', type=str,
                        default='annotations/train.json',
                        help="Anotasyon dosyasının yolu (örn: instances_val.json).")

    # Tespit dosyası için argüman
    parser.add_argument('--det_file', type=str,
                        default='neandertal.json',
                        help="Tespit sonuç dosyasının yolu (örn: result.json).")

    # Argümanları parse et
    args = parser.parse_args()

    # Değerlendirmeyi çalıştır
    map50 = eval_json(args.ann_file, args.det_file)

    # mAP@50 sonucunu yazdır
    print(f'mAP@50: {map50}')


if __name__ == "__main__":
    main()
