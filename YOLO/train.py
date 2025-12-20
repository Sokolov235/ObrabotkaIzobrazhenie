from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

results = model.train(
    data="mask_dataset.yaml",
    epochs=30,
    imgsz=320,
    batch=4,
    device='cpu',
    workers=0,
    patience=10,
    save_period=5
)