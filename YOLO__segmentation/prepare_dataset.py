import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

IMG_DIR = "MFSD/1/img"
MASK_DIR = "MFSD/1/face_crop_segmentation"

OUT_IMG_TRAIN = "dataset/images/train"
OUT_IMG_VAL = "dataset/images/val"
OUT_LBL_TRAIN = "dataset/labels/train"
OUT_LBL_VAL = "dataset/labels/val"

os.makedirs(OUT_IMG_TRAIN, exist_ok=True)
os.makedirs(OUT_IMG_VAL, exist_ok=True)
os.makedirs(OUT_LBL_TRAIN, exist_ok=True)
os.makedirs(OUT_LBL_VAL, exist_ok=True)

images = sorted(os.listdir(IMG_DIR))

train_imgs, val_imgs = train_test_split(
    images, test_size=0.2, random_state=42
)

def process_image(img_name, out_img_dir, out_lbl_dir):
    img_path = os.path.join(IMG_DIR, img_name)
    mask_path = os.path.join(MASK_DIR, img_name)

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"Пропуск файла {img_name}")
        return

    h, w = mask.shape

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    label_path = os.path.join(
        out_lbl_dir, img_name.replace(".jpg", ".txt")
    )

    with open(label_path, "w") as f:
        for cnt in contours:
            if len(cnt) < 3:
                continue

            # Класс 0 — mask
            line = "0"
            for point in cnt:
                x = point[0][0] / w
                y = point[0][1] / h
                line += f" {x:.6f} {y:.6f}"

            f.write(line + "\n")

    cv2.imwrite(os.path.join(out_img_dir, img_name), img)

for img_name in train_imgs:
    process_image(img_name, OUT_IMG_TRAIN, OUT_LBL_TRAIN)

for img_name in val_imgs:
    process_image(img_name, OUT_IMG_VAL, OUT_LBL_VAL)

print("Датасет успешно подготовлен!")