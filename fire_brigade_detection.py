import cv2
import numpy as np
import time
import os
import sys
import torch
from pathlib import Path
import pytesseract

# Set the Tesseract OCR executable path
# Update the path below if Tesseract is installed in a different location
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Assume playsound is imported from previous steps

# 1. Prepare a diverse set of pre-recorded video files
# Since we cannot load actual video files in this environment, we will simulate
# processing frames from a video. This simulation will include different scenarios
# by varying the number and type of detected vehicles, including emergency vehicles.

# Define a list of simulated video scenarios. In a real implementation,
# this would be a list of paths to actual video files.
# Each 'scenario' will be a list of simulated frames.
# Each 'frame' will contain a list of simulated detections.
# Each 'detection' will have 'box', 'score', and 'class'.

# Scenario 1: Mixed traffic, no emergency vehicles
scenario_1_frames = [
    # Frame 0
    [{'box': [100, 150, 300, 400], 'score': 0.95, 'class': 'car'},
     {'box': [450, 200, 700, 500], 'score': 0.88, 'class': 'fire brigade'},
     {'box': [50, 400, 250, 550], 'score': 0.91, 'class': 'fire brigade'}],
    # Frame 1
    [{'box': [120, 160, 320, 410], 'score': 0.94, 'class': 'car'},
     {'box': [470, 210, 720, 510], 'score': 0.87, 'class': 'fire brigade'},
     {'box': [60, 410, 260, 560], 'score': 0.90, 'class': 'fire brigade'},
     {'box': [300, 300, 400, 450], 'score': 0.85, 'class': 'motorcycle'}]
]

# Combine scenarios for simulation
simulated_video_scenarios = {
    'scenario_1_mixed_traffic': scenario_1_frames
}

# --- Define emergency vehicle classes and audio alert path ---
emergency_classes = ['fire brigade']
# Define the path to your audio alert file - assuming it exists for simulation
audio_alert_path = 'alert.mp3' # Placeholder

# Load YOLOv5 model (pre-trained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Add 'fire brigade' to the list of vehicle classes
vehicle_classes = ['car', 'bus', 'truck', 'motorcycle', 'fire brigade']

# Add a dictionary to keep track of vehicle counts
vehicle_counts = {class_name: 0 for class_name in vehicle_classes}

# Function to process video input
def process_video(video_source):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    # Ensure the camera opens in full screen mode
    cv2.namedWindow('YOLOv5 Vehicle Detection', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('YOLOv5 Vehicle Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Reset vehicle counts for the current frame
        frame_vehicle_counts = {class_name: 0 for class_name in vehicle_classes}

        # Perform detection
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()  # Get detections as numpy array

        # Lower the confidence threshold to improve detection
        CONFIDENCE_THRESHOLD = 0.3

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            class_name = model.names[int(cls)]

            # Override detected class to 'fire brigade' for bus and truck
            if class_name in ['bus', 'truck']:
                class_name = 'fire brigade'

            # Only process detections above the confidence threshold
            if conf < CONFIDENCE_THRESHOLD:
                continue

            # Only process vehicle classes
            if class_name in vehicle_classes:
                # Increment the count for the detected vehicle class
                frame_vehicle_counts[class_name] += 1

                # Draw bounding box and label
                color = (0, 255, 0) if class_name == 'fire brigade' else (255, 0, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Check if the detected class is an emergency vehicle
                if class_name in emergency_classes:
                    print(f"ALERT: Emergency vehicle detected ({class_name})!")
                    cv2.putText(frame, "EMERGENCY VEHICLE DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    cv2.imshow('YOLOv5 Vehicle Detection', frame)
                    cv2.waitKey(500)  # Pause for 500ms to emphasize the alert
                    if os.path.exists(audio_alert_path):
                        try:
                            # playsound(audio_alert_path, block=False)  # Uncomment in a real environment
                            print("(Simulating audio alert playback)")
                        except Exception as e:
                            print(f"(Simulated audio playback error: {e})")
                    else:
                        print(f"(Audio alert file not found at {audio_alert_path}. Cannot play alert.)")

        # Update cumulative vehicle counts
        for class_name, count in frame_vehicle_counts.items():
            vehicle_counts[class_name] += count

        # Display the vehicle counts on the frame
        y_offset = 30
        for class_name, count in frame_vehicle_counts.items():
            text = f"{class_name}: {count}"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 20

        # Display the frame
        cv2.imshow('YOLOv5 Vehicle Detection', frame)
        # Ensure seamless exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting program...")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print cumulative vehicle counts
    print("Cumulative Vehicle Counts:")
    for class_name, count in vehicle_counts.items():
        print(f"{class_name}: {count}")

# Main function
def main():
    video_source = 0  # Use 0 for webcam or provide a video file path
    process_video(video_source)

if __name__ == "__main__":
    main()