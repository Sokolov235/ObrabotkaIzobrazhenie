from ultralytics import YOLO

model = YOLO('runs/segment/train/weights/best.pt')

results = model.val()

results = model.predict('test.jpg', save=True, show=True)