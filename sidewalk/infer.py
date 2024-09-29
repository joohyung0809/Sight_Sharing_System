from ultralytics import YOLO

model = YOLO("./model/best.pt")


results = model(source="./5.mp4",
                save=True,
                show_boxes=False,
                show_conf=True,
                conf=0.8)
