from ultralytics import YOLO

model = YOLO("./runs/detect/train6/weights/best.pt")

success = model.export(format="onnx", opset=12, simplify=True)