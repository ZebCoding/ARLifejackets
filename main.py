import cv2
import numpy as np
import mediapipe as mp

# Load lifejacket with alpha channel
lifejacket = cv2.imread('lifejacket.png', cv2.IMREAD_UNCHANGED)
if lifejacket is None:
    print("Error: lifejacket.png not found")
    exit()

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(1)  # Change index if needed
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Output video
out = cv2.VideoWriter('output.avi',
                      cv2.VideoWriter_fourcc(*'MJPG'),
                      15.0,
                      (640, 480))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Get torso keypoints
        left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

        # Convert normalized coordinates to pixels
        x1 = int(left_shoulder.x * frame.shape[1])
        x2 = int(right_shoulder.x * frame.shape[1])
        y1 = int(left_shoulder.y * frame.shape[0])
        y2 = int(((left_hip.y + right_hip.y)/2) * frame.shape[0])

        # Ensure valid rectangle
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        width = x2 - x1
        height = y2 - y1

        if width > 0 and height > 0:
            # Resize lifejacket
            resized_lifejacket = cv2.resize(lifejacket, (width, height))

            # Handle alpha channel
            if resized_lifejacket.shape[2] == 4:
                b, g, r, a = cv2.split(resized_lifejacket)
                overlay_color = cv2.merge((b, g, r))
                mask = a
            else:
                overlay_color = resized_lifejacket
                mask = 255 * np.ones(resized_lifejacket.shape[:2], dtype=np.uint8)

            # Resize mask and ROI to match
            mask = cv2.resize(mask, (overlay_color.shape[1], overlay_color.shape[0]))
            mask_inv = cv2.bitwise_not(mask)

            roi = frame[y1:y2, x1:x2]
            roi = cv2.resize(roi, (overlay_color.shape[1], overlay_color.shape[0]))

            roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            overlay_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)
            combined = cv2.add(roi_bg, overlay_fg)

            # Put back into frame
            h_frame, w_frame = frame.shape[:2]
            x_end = min(x1 + combined.shape[1], w_frame)
            y_end = min(y1 + combined.shape[0], h_frame)

            # Adjust combined size if it exceeds frame
            combined_cropped = combined[:y_end - y1, :x_end - x1]
            # Only overlay if valid
            if combined_cropped.shape[0] > 0 and combined_cropped.shape[1] > 0:
                frame[y1:y_end, x1:x_end] = combined_cropped

    # Write output video and show frame
    out.write(frame)
    cv2.imshow('AR Lifejacket', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
