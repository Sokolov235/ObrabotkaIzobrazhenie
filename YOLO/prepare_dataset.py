import os
import cv2
import random
from tqdm import tqdm

SRC_IMG = "MFSD/1/img"
SRC_MASK = "MFSD/1/face_crop_segmentation"

DST_IMG_TRAIN = "MFSD/images/train"
DST_IMG_VAL = "MFSD/images/val"
DST_LAB_TRAIN = "MFSD/labels/train"
DST_LAB_VAL = "MFSD/labels/val"

os.makedirs(DST_IMG_TRAIN, exist_ok=True)
os.makedirs(DST_IMG_VAL, exist_ok=True)
os.makedirs(DST_LAB_TRAIN, exist_ok=True)
os.makedirs(DST_LAB_VAL, exist_ok=True)

files = [f for f in os.listdir(SRC_IMG) if f.endswith(".jpg")]
random.shuffle(files)

files = files[:1000]  # берем только 1000 изображений, потому что нет места на пк)

split = int(0.8 * len(files))
train_files = files[:split]
val_files = files[split:]


def find_mask(img_name):
    base = img_name.replace(".jpg", "")
    for f in os.listdir(SRC_MASK):
        if f.startswith(base + "_") and f.endswith(".jpg"):
            return os.path.join(SRC_MASK, f)
    return None


def process(file, img_dst, lab_dst):
    img_path = os.path.join(SRC_IMG, file)
    mask_path = find_mask(file)

    if mask_path is None:
        return

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)

    h, w = mask.shape
    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return

    cnt = max(contours, key=cv2.contourArea)

    points = []
    for p in cnt:
        x = p[0][0] / w
        y = p[0][1] / h
        points.append(f"{x:.6f} {y:.6f}")

    label_line = "0 " + " ".join(points)

    cv2.imwrite(os.path.join(img_dst, file), img)
    with open(os.path.join(lab_dst, file.replace(".jpg", ".txt")), "w") as f:
        f.write(label_line)


print("Processing train...")
for f in tqdm(train_files):
    process(f, DST_IMG_TRAIN, DST_LAB_TRAIN)

print("Processing val...")
for f in tqdm(val_files):
    process(f, DST_IMG_VAL, DST_LAB_VAL)

print("Done!")