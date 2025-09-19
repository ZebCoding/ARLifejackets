import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from scipy.spatial import distance

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt")  # small, fast model

# Load lifejacket with alpha channel
lifejacket = cv2.imread('lifejacket.png', cv2.IMREAD_UNCHANGED)
if lifejacket is None:
    print("Error: lifejacket.png not found")
    exit()

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(1)  # use your webcam index
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Store smoothed coordinates for each person
prev_coords = {}  # key = person_id, value = dict with x_start, y_start, x_end, y_end
alpha = 0.3  # smoothing factor

def match_person(x_center, y_center, prev_coords):
    """Match current detected person to previous coords by proximity"""
    for pid, coords in prev_coords.items():
        px_center = (coords['x_start'] + coords['x_end']) / 2
        py_center = (coords['y_start'] + coords['y_end']) / 2
        if distance.euclidean((x_center, y_center), (px_center, py_center)) < 50:  # threshold
            return pid
    return None

person_counter = 0  # unique ID counter

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect all people in the frame
    results = model(frame, verbose=False)
    boxes = results[0].boxes

    current_coords = {}

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            if int(box.cls[0]) == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_person = frame[y1:y2, x1:x2]
                if cropped_person.size == 0:
                    continue

                # Run Mediapipe Pose
                rgb_crop = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(rgb_crop)

                if pose_results.pose_landmarks:
                    lm = pose_results.pose_landmarks.landmark

                    # Torso keypoints
                    left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                    right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                    left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
                    right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]

                    # Map to original frame
                    ls_x = int(left_shoulder.x * (x2 - x1)) + x1
                    rs_x = int(right_shoulder.x * (x2 - x1)) + x1
                    ls_y = int(left_shoulder.y * (y2 - y1)) + y1
                    rs_y = int(right_shoulder.y * (y2 - y1)) + y1
                    lh_y = int(left_hip.y * (y2 - y1)) + y1
                    rh_y = int(right_hip.y * (y2 - y1)) + y1

                    # Rectangle for lifejacket
                    x_start = min(ls_x, rs_x)
                    x_end = max(ls_x, rs_x)
                    y_start = min(ls_y, rs_y)
                    y_end = max(lh_y, rh_y)

                    # Determine person ID for smoothing
                    x_center = (x_start + x_end) / 2
                    y_center = (y_start + y_end) / 2
                    person_id = match_person(x_center, y_center, prev_coords)
                    if person_id is None:
                        person_id = person_counter
                        person_counter += 1

                    # Apply exponential smoothing
                    if person_id in prev_coords:
                        x_start = int(alpha * x_start + (1 - alpha) * prev_coords[person_id]['x_start'])
                        x_end   = int(alpha * x_end   + (1 - alpha) * prev_coords[person_id]['x_end'])
                        y_start = int(alpha * y_start + (1 - alpha) * prev_coords[person_id]['y_start'])
                        y_end   = int(alpha * y_end   + (1 - alpha) * prev_coords[person_id]['y_end'])

                    # Save smoothed coordinates
                    current_coords[person_id] = {'x_start': x_start, 'x_end': x_end, 'y_start': y_start, 'y_end': y_end}

                    # Width and height safety check
                    width = max(1, x_end - x_start)
                    height = max(1, y_end - y_start)

                    # Skip overlay if size invalid
                    if width <= 0 or height <= 0:
                        continue

                    # Resize lifejacket
                    resized_lifejacket = cv2.resize(lifejacket, (width, height), interpolation=cv2.INTER_AREA)

                    # Alpha blending
                    if resized_lifejacket.shape[2] == 4:
                        b, g, r, a = cv2.split(resized_lifejacket)
                        overlay_color = cv2.merge((b, g, r))
                        alpha_mask = a / 255.0
                    else:
                        overlay_color = resized_lifejacket
                        alpha_mask = np.ones(resized_lifejacket.shape[:2], dtype=float)

                    # Overlay onto frame
                    y1_overlay = y_start
                    y2_overlay = y_start + overlay_color.shape[0]
                    x1_overlay = x_start
                    x2_overlay = x_start + overlay_color.shape[1]

                    # Clamp to frame
                    y1_overlay = max(0, y1_overlay)
                    x1_overlay = max(0, x1_overlay)
                    y2_overlay = min(frame.shape[0], y2_overlay)
                    x2_overlay = min(frame.shape[1], x2_overlay)

                    roi = frame[y1_overlay:y2_overlay, x1_overlay:x2_overlay]
                    if roi.shape[0] == 0 or roi.shape[1] == 0:
                        continue

                    # Resize alpha mask if needed
                    alpha_resized = cv2.resize(alpha_mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_AREA)
                    overlay_resized = cv2.resize(overlay_color, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_AREA)

                    for c in range(3):
                        roi[:, :, c] = (alpha_resized * overlay_resized[:, :, c] +
                                        (1 - alpha_resized) * roi[:, :, c]).astype(np.uint8)

                    frame[y1_overlay:y2_overlay, x1_overlay:x2_overlay] = roi

    # Update previous coords for next frame (safe even if nobody detected)
    prev_coords = current_coords

    # Show the frame
    cv2.imshow('AR Lifejacket Overlay (Smoothed & Safe)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
