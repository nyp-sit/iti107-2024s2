from ultralytics import YOLO

model = YOLO("yolo11s.pt", task="detection")
model.predict(source="balloon.mp4", stream=True, show=True)

