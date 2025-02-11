import threading
import cv2
import numpy as np
import os
import requests
from datetime import datetime

# Thread-safe last_alert_time dictionary
last_alert_time = {}
alert_time_lock = threading.Lock()

# Load VIT campus map
file_path = r"C:\Users\TrackBee\Downloads\VIT-MAP.png"

# Check if the file exists
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit()

map_image = cv2.imread(file_path)
if map_image is None:
    print("Error: Unable to load the map image. Check the file path and format.")
    exit()

# Define multiple camera locations
camera_locations = [
    {"name": "Canteen", "coords": (600, 400), "density": "green", "camera_index": 0},
    {"name": "Library", "coords": (300, 200), "density": "green", "camera_index": 1}
]

# Global frames dictionary for each camera
frames = {cam["name"]: None for cam in camera_locations}
frame_locks = {cam["name"]: threading.Lock() for cam in camera_locations}

# Define density colors
density_colors = {
    "green": (0, 255, 0, 128),  # Low density
    "orange": (0, 165, 255, 128),  # Medium density
    "red": (0, 0, 255, 128)  # High density
}


# Load class labels for COCO dataset
def load_classes(file_path):
    with open(file_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


# Load YOLO model
weights_path = r"C:\Users\TrackBee\Downloads\yolov4.weights"
config_path = r"C:\Users\TrackBee\Downloads\yolov4.cfg"
coco_names_path = r"C:\Users\TrackBee\Downloads\coco.names"

# Check if the YOLO files exist
if not os.path.exists(weights_path) or not os.path.exists(config_path) or not os.path.exists(coco_names_path):
    print(
        "YOLO files not found. Please ensure yolov4.weights, yolov4.cfg, and coco.names are in the correct directory.")
    exit()

# Load YOLO
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
classes = load_classes(coco_names_path)


# Function to send an alert
def send_alert_to_twilio(camera_name, issue):
    current_time = datetime.now()
    with alert_time_lock:
        if camera_name in last_alert_time:
            time_diff = current_time - last_alert_time[camera_name]
            if time_diff.total_seconds() < 1200:
                return
        last_alert_time[camera_name] = current_time
    print(f"Alert for {camera_name}: {issue}")


# Function to capture frames for each camera
def capture_frames(camera_name, camera_index):
    global frames
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Failed to open camera {camera_name} at index {camera_index}")
        return
    while True:
        ret, new_frame = cap.read()
        if ret:
            with frame_locks[camera_name]:
                frames[camera_name] = new_frame


# Start threads for each camera
for camera in camera_locations:
    threading.Thread(
        target=capture_frames,
        args=(camera["name"], camera["camera_index"]),
        daemon=True
    ).start()


# Function to detect fighting motion in optical flow
def is_fighting_motion(flow_region):
    # Placeholder for actual logic to detect fighting motion
    return np.mean(flow_region) > 2.5  # Example threshold


# Function to process frames for crowd density and objects
def process_frame(frame, net, output_layers, classes, prev_gray=None):
    # Reduce resolution to improve performance
    frame_resized = cv2.resize(frame, (640, 480))

    # Run object detection on the resized frame
    blob = cv2.dnn.blobFromImage(frame_resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    height, width, _ = frame_resized.shape
    bboxes, confidences, class_ids = [], [], []
    people_count = 0
    other_objects = {}

    # Convert the resized frame to grayscale
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # If prev_gray exists, calculate optical flow
    if prev_gray is not None:
        # Ensure the previous and current frames are the same size
        if prev_gray.shape != gray.shape:
            prev_gray = cv2.resize(prev_gray, (gray.shape[1], gray.shape[0]))

        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    else:
        flow = None  # First frame, no flow calculation

    # Process object detections using YOLO
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.9:  # Adjust from 0.5 to a higher value

                label = classes[class_id]
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                bboxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                if label == "person":
                    people_count += 1
                    if flow is not None:
                        # Analyze optical flow around the person's bounding box
                        flow_region = flow[y:y + h, x:x + w]

                        if is_fighting_motion(flow_region):
                            send_alert_to_twilio("Camera", "Fighting detected")
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                else:
                    other_objects[label] = other_objects.get(label, 0) + 1

    return people_count, other_objects, bboxes, gray  # Return prev_gray (gray) for the next frame


# Main loop
prev_gray = None  # Initialize prev_gray for the first frame
while True:
    map_with_overlay_copy = map_image.copy()
    for camera in camera_locations:
        with frame_locks[camera["name"]]:
            if frames[camera["name"]] is not None:
                people_count, other_objects, _, prev_gray = process_frame(
                    frames[camera["name"]], net, output_layers, classes, prev_gray
                )

                # Determine the density color based on people_count
                if people_count < 2:
                    density_color = density_colors["green"]
                elif 2 <= people_count < 3:
                    density_color = density_colors["orange"]
                else:
                    density_color = density_colors["red"]

                # Add density circle to the map
                cv2.circle(map_with_overlay_copy, camera["coords"], 50, density_color[:3], -1)
                cv2.putText(map_with_overlay_copy, f"People: {people_count}",
                            (camera["coords"][0] - 80, camera["coords"][1] + 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                # Alert for detected weapons or other critical objects
                if other_objects:
                    send_alert_to_twilio(camera["name"], f"Objects detected: {other_objects}")

                # Show the frame feed
                cv2.imshow(f"Camera Feed - {camera['name']}", frames[camera["name"]])

    cv2.imshow("Campus Map with Density", map_with_overlay_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
