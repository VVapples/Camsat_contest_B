import cv2
import apriltag # This will now use the 'pupil-apriltags' library
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os

# Open a file dialog to select the video file
root = tk.Tk()
root.withdraw()  # Hide the main window

# Try to get the script directory, fall back to current working directory
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd() # Fallback for interactive environments

video_path = filedialog.askopenfilename(
    initialdir=script_dir,
    title="Select Video File",
    filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
)

# Check if a file was selected
if not video_path:
    print("No video file selected. Exiting.")
    exit()

output_csv = "tag_drift_analysis.csv"

# Initialize AprilTag detector for the 'tag36h11' family
# --- THIS IS THE MODIFIED LINE ---
detector = apriltag.Detector(families='tag36h11') 

cap = cv2.VideoCapture(video_path)
frame_index = 0
tag_positions = []

print("Processing video frames...")
# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    for det in detections:
        tag_id = det.tag_id
        center = det.center
        tag_positions.append({
            "frame": frame_index,
            "tag_id": tag_id,
            "x": center[0],
            "y": center[1]
        })

    frame_index += 1

cap.release()
print("Video processing complete.")

# Create DataFrame and compute drift
if tag_positions:
    df = pd.DataFrame(tag_positions)
    # Ensure dataframe is sorted by tag_id then frame to calculate diff correctly
    df = df.sort_values(by=['tag_id', 'frame'])
    
    df["dx"] = df.groupby("tag_id")["x"].diff()
    df["dy"] = df.groupby("tag_id")["y"].diff()
    df["drift"] = np.sqrt(df["dx"]**2 + df["dy"]**2)

    # Save results to CSV
    df.to_csv(output_csv, index=False)
    print(f"Drift analysis completed. Results saved to {output_csv}")
else:
    print("No AprilTags were detected in the video.")

