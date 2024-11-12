import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import sqlite3
from datetime import datetime

# Load the YOLOv8 model
model = YOLO("../weights/yolov8x.pt")  # detection

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

# Connect to the SQLite database
conn = sqlite3.connect('triple_s.db')
cursor = conn.cursor()

# Function to calculate movement direction (left or right)
def calculate_direction(start_point, end_point):
    if end_point[0] < start_point[0]:
        return 'left'
    else:
        return 'right'

while cap.isOpened():
    # Read a frame from the webcam
    success, frame = cap.read()

    if success:
        # Run YOLOv8 detection & tracking on the frame
        results = model.track(frame, persist=True, show_conf=False, classes=0)  # Detect only 'person'

        # boxes와 track_ids를 초기화
        boxes = []
        track_ids = []

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xywh.cpu()  # Bounding boxes (xywh format)
            
            # track_ids 추출할 때 None 체크
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs


        if len(boxes) > 0:
            # Calculate area of each bounding box (width * height)
            areas = [(w * h, idx) for idx, (x, y, w, h) in enumerate(boxes)]
            # Get the index of the bounding box with the largest area
            _, max_area_idx = max(areas, key=lambda item: item[0])

            # Extract the largest bounding box
            largest_box = boxes[max_area_idx]

            # Check if track_ids is not empty and has sufficient length
            if len(track_ids) > max_area_idx:
                largest_track_id = track_ids[max_area_idx]
            else:
                # If track_ids is empty or shorter, set a default value (e.g., -1)
                largest_track_id = -1

            # Visualize bounding boxes
            for idx, (box, track_id) in enumerate(zip(boxes, track_ids)):
                x, y, w, h = box
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)

                # Check if the current box is the largest one
                if idx == max_area_idx:
                    # Draw the largest bounding box in red
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red color
                else:
                    # Draw other bounding boxes in blue
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue color

            # Track and analyze the movement of the largest object
            x, y, w, h = largest_box
            track = track_history[largest_track_id]
            current_point = (float(x), float(y))  # Current center point
            track.append(current_point)

            # Initialize a flag for danger detection
            danger_flag = False

            # If track has more than 5 points, retain the last 30 points and analyze direction
            if len(track) > 6:
                track.pop(0)

                # Draw the tracking line
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

                # Draw an arrow based on the last 5 frames
                start_point = tuple(map(int, track[-5]))
                end_point = tuple(map(int, track[-1]))

                # Check if the movement direction is towards the left
                if end_point[0] < start_point[0]:
                    danger_flag = True

                # Draw the arrowed line indicating the movement direction
                cv2.arrowedLine(frame, start_point, end_point, color=(0, 255, 0), thickness=3, tipLength=0.3)

                # Calculate direction (left or right)
                direction = calculate_direction(start_point, end_point)

                # Insert data into the database (without risk_level)
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute(
                    """
                    INSERT INTO detections 
                    (timestamp, object_type, x_coordinate, y_coordinate, direction) 
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (current_time, 'person', x.item(), y.item(), direction)
                )
                conn.commit()

            # Display "DANGER" if the largest object is moving to the left
            if danger_flag:
                cv2.putText(
                    frame,
                    "DANGER",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 255),  # Red color for danger
                    3,
                    cv2.LINE_AA,
                )

            # Display the frame with the bounding boxes and tracking information
            cv2.imshow("YOLOv8 Tracking with Webcam Input", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if frame reading fails
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
conn.close()
