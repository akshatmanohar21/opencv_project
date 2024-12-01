import cv2
import pickle
import cvzone
import numpy as np
import datetime
import json
import csv

# Load video and parking space positions
cap = cv2.VideoCapture('carPark.mp4')
width, height = 103, 43

# Load parking space positions
try:
    with open('CarParkPos', 'rb') as f:
        posList = pickle.load(f)
except FileNotFoundError:
    print("Error: CarParkPos file not found. Run ParkingSpacePicker.py to create it.")
    exit()

# Initialize YOLO model (using YOLOv4-tiny for speed)
net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)  # Enable GPU acceleration
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Get layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize space tracking
space_time = {idx: 0 for idx in range(len(posList))}  # Tracks occupancy duration
space_start_time = {idx: None for idx in range(len(posList))}  # Start time for occupancy
space_previous_state = {idx: "Free" for idx in range(len(posList))}  # Tracks previous state

# Vehicle type mapping (COCO class IDs)
vehicle_types = {2: "Car", 3: "Motorcycle", 7: "Truck"}

# Initialize log data
log_data = []

# Logging function
def log_parking_status(space_id, status, vehicle_type=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "timestamp": timestamp,
        "space_id": space_id,
        "status": status,
        "vehicle_type": vehicle_type
    }
    log_data.append(entry)
    with open("parking_log.json", "w") as json_file:
        json.dump(log_data, json_file, indent=4)

    with open("parking_log.csv", "a") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([timestamp, space_id, status, vehicle_type])

# Heatmap generation
heatmap = np.zeros((720, 1280), dtype=np.float32)  # Example dimensions for heatmap

# Function to apply heatmap overlay
def apply_heatmap(img):
    global heatmap
    # Normalize the heatmap
    normalized_heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the heatmap to a color image
    colored_heatmap = cv2.applyColorMap(normalized_heatmap.astype(np.uint8), cv2.COLORMAP_JET)

    # Resize the heatmap to match the input frame dimensions
    colored_heatmap = cv2.resize(colored_heatmap, (img.shape[1], img.shape[0]))

    # Blend the heatmap with the input image
    overlay = cv2.addWeighted(img, 0.7, colored_heatmap, 0.3, 0)
    return overlay


# Function to check parking spaces
def checkSpaces(img, imgThres):
    spaces_free = 0  # Count free spaces

    for idx, pos in enumerate(posList):
        x, y = pos
        w, h = width, height

        # Crop the parking space area
        imgCropThres = imgThres[y:y + h, x:x + w]
        count = cv2.countNonZero(imgCropThres)

        # Determine if the parking space is free or occupied
        if count < 900:
            color = (0, 200, 0)
            label = "Free"
            spaces_free += 1

            # Reset occupancy time
            space_time[idx] = 0
            space_start_time[idx] = None
        else:
            color = (0, 0, 200)
            label = "Occupied"
            heatmap[y:y + h, x:x + w] += 1  # Update heatmap for occupied space

            if space_start_time[idx] is None:  # Space just became occupied
                space_start_time[idx] = datetime.datetime.now()

            # Calculate occupancy duration
            if space_start_time[idx] is not None:
                duration = datetime.datetime.now() - space_start_time[idx]
                minutes = int(duration.total_seconds() // 60)
                seconds = int(duration.total_seconds() % 60)
                duration_text = f'{minutes}m {seconds}s'
                cv2.putText(img, duration_text, (x + 5, y + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Log state changes
        if space_previous_state[idx] != label:
            log_parking_status(idx + 1, label)  # Log when state changes
            space_previous_state[idx] = label  # Update the previous state

        # Draw bounding boxes and labels
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f'{count}', (x, y + h - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
        cv2.putText(img, label, (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Display occupancy percentage
    occupancy_percentage = (len(posList) - spaces_free) / len(posList) * 100
    cvzone.putTextRect(img, f'Free: {spaces_free}/{len(posList)}', (50, 60), thickness=1, offset=20, colorR=(0, 200, 0))
    cvzone.putTextRect(img, f'Occupancy: {occupancy_percentage:.2f}%', (50, 100), thickness=1, offset=20, colorR=(255, 255, 0))

# Main loop
frame_skip = 2  # Process every 2nd frame to reduce computational load
frame_count = 0

while True:
    success, img = cap.read()

    # If the video ends, loop back to the beginning
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Skip frames for performance optimization
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # YOLO Detection
    height_img, width_img = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    for detection in detections:
        for object_detection in detection:
            scores = object_detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Detect only vehicles (Car, Motorcycle, Truck)
            if confidence > 0.5 and class_id in vehicle_types:
                vehicle_label = vehicle_types[class_id]
                center_x = int(object_detection[0] * width_img)
                center_y = int(object_detection[1] * height_img)
                w = int(object_detection[2] * width_img)
                h = int(object_detection[3] * height_img)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw rectangle and label
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(img, vehicle_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # Convert to grayscale and apply Canny edge detection
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgEdges = cv2.Canny(imgGray, 50, 150)

    # Apply morphological transformations for noise reduction
    kernel = np.ones((3, 3), np.uint8)
    imgThres = cv2.morphologyEx(imgEdges, cv2.MORPH_CLOSE, kernel)

    # Check parking spaces
    checkSpaces(img, imgThres)

    # Apply heatmap overlay
    imgWithHeatmap = apply_heatmap(img)

    # Display the results
    cv2.imshow("Parking Detection with Vehicle Types", imgWithHeatmap)

    # Exit on pressing 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
