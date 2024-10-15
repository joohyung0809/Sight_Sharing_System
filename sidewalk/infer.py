from ultralytics import YOLO

model = YOLO("../model/roadway.pt")


results = model(source="../vid/5.mp4", save=True, show=True)
