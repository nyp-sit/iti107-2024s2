from ultralytics import YOLO

model = YOLO("yolo11s.pt", task="detection")
model.predict(source=0, show=True)

