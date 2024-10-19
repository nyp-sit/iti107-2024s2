from ultralytics.utils.benchmarks import benchmark
from ultralytics import YOLO

model = YOLO('best.pt')
model.export(format="openvino")
# # Benchmark on GPU
# benchmark(model="best.pt", data="datasets/data.yaml", imgsz=640, half=False, device=0)
