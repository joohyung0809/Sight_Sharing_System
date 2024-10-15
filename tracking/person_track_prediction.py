import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Load the YOLOv8 models
model = YOLO("../weights/yolov8n.pt")  # detection
model2 = YOLO("../model/roadway.pt")  # segmentation

# Open the webcam (use '0' for the default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get the webcam's width, height, and frames per second (fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the webcam frames
while cap.isOpened():
    # Read a frame from the webcam
    success, frame = cap.read()

    if success:
        # Run YOLOv8 detection & tracking on the frame
        results = model.track(frame, persist=True, classes=0)  # det&track
        results2 = model2(frame, show_boxes=False, show_conf=True)  # segmentation

        # Create a copy of the original frame for drawing results
        frame_with_edges = frame.copy()

        # Check if segmentation masks are available
        if results2[0].masks is not None:
            # Get the segmentation mask
            seg_mask = results2[0].masks.data[0].cpu().numpy()  # Mask for the first segmented object
            seg_mask = cv2.resize(seg_mask, (width, height))  # Resize to match frame size

            # Extract edges from the segmentation mask
            edges = cv2.Canny((seg_mask * 255).astype(np.uint8), 50, 150)  # Canny edge detection

            # Define a kernel for dilation (to make edges thicker)
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

            # Draw the extracted edges in deep blue
            frame_with_edges[edges != 0] = [139, 0, 0]  # Set edge pixels to deep blue

        # Check if any boxes (detections) are available before proceeding
        if results[0].boxes is not None and results[0].boxes.id is not None:
            # Get the boxes and track IDs from the detection
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Draw the tracking lines and arrows for each tracked object
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                current_point = (float(x), float(y))  # Current center point
                track.append(current_point)

                if len(track) > 20:  # If track has more than 20 points, retain last 20 and draw arrow
                    track.pop(0)

                    # Draw the tracking lines
                    points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame_with_edges, [points], isClosed=False, color=(230, 230, 230), thickness=2)

                    # Draw an arrow based on the last frames
                    start_point = track[-20]
                    end_point = track[-1]

                    start_point = tuple(map(int, start_point))
                    end_point = tuple(map(int, end_point))

                    # Draw the arrowed line indicating the movement direction
                    cv2.arrowedLine(frame_with_edges, start_point, end_point, color=(0, 255, 0), thickness=3, tipLength=0.3)

        # Show the frame with the overlaid edges (if available) and tracking information
        cv2.imshow("YOLOv8 Tracking with Edges Overlay (Webcam)", frame_with_edges)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if frame reading fails
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
