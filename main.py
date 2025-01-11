from gaze_tracking import GazeTracking
from pynput.mouse import Controller
import cv2
import numpy as np
import tkinter as tk

# Function to get the screen resolution
def get_screen_resolution():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    return width, height

# Fetch screen resolution dynamically
screen_width, screen_height = get_screen_resolution()

# Initialize gaze tracker and mouse controller
gaze = GazeTracking()
mouse = Controller()

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not detected.")
    exit()

# Initialize previous gaze position for calculating deltas
prev_horizontal_ratio, prev_vertical_ratio = None, None

# Movement smoothing and sensitivity
time_constant = 3.0  # Adjustable time constant for the filter (in seconds)
sensitivity = 4  # Sensitivity coefficient (adjust for faster/slower movement)
filtered_delta_x, filtered_delta_y = 0, 0  # Filtered deltas

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Analyze gaze direction
    gaze.refresh(frame)
    frame = gaze.annotated_frame()

    # Get horizontal and vertical gaze ratio
    horizontal_ratio = gaze.horizontal_ratio()  # 0 (left) to 1 (right)
    vertical_ratio = gaze.vertical_ratio()      # 0 (top) to 1 (bottom)

    if horizontal_ratio is not None and vertical_ratio is not None:
        # Calculate deltas if previous position exists
        if prev_horizontal_ratio is not None and prev_vertical_ratio is not None:
            delta_x = (horizontal_ratio - prev_horizontal_ratio) * screen_width * sensitivity
            delta_y = (vertical_ratio - prev_vertical_ratio) * screen_height * sensitivity

            # Apply smoothing using a first-order filter
            alpha = 1 - np.exp(-1 / time_constant)  # Smoothing factor
            filtered_delta_x = alpha * delta_x + (1 - alpha) * filtered_delta_x
            filtered_delta_y = alpha * delta_y + (1 - alpha) * filtered_delta_y

            # Move the mouse incrementally
            current_position = mouse.position
            mouse.position = (
                max(0, min(screen_width - 1, int(current_position[0] - filtered_delta_x))),
                max(0, min(screen_height - 1, int(current_position[1] + filtered_delta_y)))
            )

        # Update previous gaze position
        prev_horizontal_ratio = horizontal_ratio
        prev_vertical_ratio = vertical_ratio

    # Display the video feed with gaze tracking annotations
    cv2.imshow("Gaze Tracking", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()