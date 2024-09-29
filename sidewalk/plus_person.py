from ultralytics import YOLO
import cv2

# Load your primary model (e.g., custom trained model)
primary_model = YOLO("./model/best.pt")

# Load the pedestrian detection model (e.g., YOLOv5 trained on COCO)
pedestrian_model = YOLO("yolov8n.pt")  # Ensure this model is trained to detect pedestrians

# Capture video feed and apply the detection models
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video feed
    ret, frame = cap.read()
    if not ret:
        break
    
    results0 = primary_model(source=frame, show=False)
    results1 = pedestrian_model(source=frame, classes=0)

    if results0.masks is not None:
        masks = results0.masks
        for mask in masks:
            # Convert mask to an 8-bit binary image
            mask_img = (mask * 255).astype(np.uint8)
            # Apply a colormap (e.g., COLORMAP_JET) to the mask image
            colored_mask = cv2.applyColorMap(mask_img, cv2.COLORMAP_JET)
            # Blend the mask with the original frame
            output_frame = cv2.addWeighted(output_frame, 1.0, colored_mask, 0.5, 0)

    # Draw bounding boxes (from results1)
    if results1.boxes is not None:
        boxes = results1.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a green rectangle

    # Display the frame with detections
    cv2.imshow("Detections", output_frame)
    

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
