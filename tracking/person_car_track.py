from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("../weights/yolov8n.pt")  # detection
model2 = YOLO("../weights/best.pt")  # segmentation

# Open the video file
video_path = "../vid/people wlaking.mp4"
cap = cv2.VideoCapture(video_path)

# Get the video's width, height, and frames per second (fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object to save the video
output_path = "./runs/output_with_tracking.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 detection & tracking on the frame
        results = model.track(frame, persist=True, classes=0, save=True)  # det&track
        results2 = model2(frame, save=True, show_boxes=False, show_conf=True, conf=0.8)  # segmentation

        # Get the segmentation mask (assuming it's the first mask)
        seg_mask = results2[0].masks.data[0].cpu().numpy()  # Mask for the first segmented object (e.g., the sidewalk)
        seg_mask = cv2.resize(seg_mask, (width, height))  # Resize to match frame size

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks and arrows
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            current_point = (float(x), float(y))  # Current center point
            track.append(current_point)

            # If track has more than 5 points, retain 30 and draw the arrow
            if len(track) > 5:
                track.pop(0)

                # Draw the tracking lines
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

                # Draw an arrow based on the last 5 frames
                start_point = track[-5]  # Five frames ago
                end_point = track[-1]  # Current frame

                # Convert to integer points
                start_point = tuple(map(int, start_point))
                end_point = tuple(map(int, end_point))

                # Draw the arrowed line indicating the movement direction
                cv2.arrowedLine(annotated_frame, start_point, end_point, color=(0, 255, 0), thickness=3, tipLength=0.3)

                # Check if the person is moving out of the sidewalk (segmentation mask area)
                # Assuming '1' in seg_mask represents the sidewalk area
                if seg_mask[int(end_point[1]), int(end_point[0])] == 0:  # Outside the sidewalk
                    # Display a warning on the frame
                    cv2.putText(annotated_frame, "Danger!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    # Print "Danger" in the terminal
                    print(f"Warning! Track ID {track_id} is moving outside the sidewalk.")

        # Save the frame to the output video
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking with Direction Arrows and Danger Warning", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture and writer objects, and close the display window
cap.release()
out.release()  # Ensure the output video file is properly saved
cv2.destroyAllWindows()
