import cv2
import numpy as np
from pynput.mouse import Controller

# Initialize mouse controller
mouse = Controller()

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not detected.")
    exit()

# Load pre-trained Haar cascades for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    for (x, y, w, h) in eyes[:1]:  # Focus on one eye
        # Get the center of the eye
        eye_center = (x + w // 2, y + h // 2)

        # Map the eye position to screen size
        screen_width, screen_height = 1920, 1080  # Adjust to your screen size
        cam_width, cam_height = cap.get(3), cap.get(4)

        mapped_x = int(eye_center[0] * screen_width / cam_width)*0.8
        mapped_y = int(eye_center[1] * screen_height / cam_height)*0.8

        # Move the mouse
        mouse.position = (mapped_x, mapped_y)

        # Draw a rectangle around the eye
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Eye Tracker', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()