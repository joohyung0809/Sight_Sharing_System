from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)

results = model.train(data="sidewalk.yaml", epochs=100, imgsz=640)
