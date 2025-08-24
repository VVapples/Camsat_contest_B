# -*- coding: utf-8 -*-
"""
Detects AprilTags in a video file and saves the tracking data with timestamps.

This script is designed to work within the project's file structure.

Standalone Mode:
- Run `python tracking.py`.
- A file dialog will open to select a video.

Pipeline Mode (called from main.py):
- Run `python tracking.py "path/to/video.mp4"`.
- The specified video will be processed directly.

The script automatically finds the corresponding calibration file from
`Outputs/Calibration/` and saves the output CSV to
`Outputs/Tracking/(cameraname)/(videoname)_data.csv`.
"""
import cv2
import numpy as np
import os
import sys
import tkinter as tk
from tkinter import filedialog
from pupil_apriltags import Detector
import csv

# --- Configuration ---
TAG_FAMILY = 'tag36h11'
TAG_SIZE_M = 0.055 # The size of your AprilTag in meters.

def track_apriltags_in_video(video_path):
    """
    Processes a single video file to detect AprilTags and save their data.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        sys.exit(1)

    # --- 1. Automatically Find and Load Calibration Data ---
    try:
        # Get camera name from the parent folder of the video
        folder_path = os.path.dirname(video_path)
        camera_name = os.path.basename(folder_path)
        calibration_path = os.path.join("Outputs", "Calibration", f"{camera_name}.npz")

        if not os.path.exists(calibration_path):
            print(f"Error: Calibration file not found at expected path: {calibration_path}")
            print("Please run the camera calibration for this camera first.")
            sys.exit(1)

        with np.load(calibration_path) as data:
            mtx, dist = data['mtx'], data['dist']
        print(f"Successfully loaded calibration data for '{camera_name}'")
    except Exception as e:
        print(f"Error loading calibration file: {e}")
        sys.exit(1)

    # --- 2. Initialize AprilTag Detector ---
    camera_params = [mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2]] # fx, fy, cx, cy
    at_detector = Detector(
        families=TAG_FAMILY, nthreads=1, quad_decimate=1.5, quad_sigma=0.8,
        refine_edges=1, decode_sharpening=0.25, debug=0
    )

    # --- 3. Prepare Video Input and Output ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_dir = os.path.join("Outputs", "Tracking", camera_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_filename = os.path.splitext(os.path.basename(video_path))[0]
    output_csv_path = os.path.join(output_dir, f"{base_filename}_data.csv")
    
    # --- 4. Setup CSV File for Data Logging ---
    try:
        csv_file = open(output_csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        
        header = ['frame', 'timestamp_ms', 'tag_id', 'center_x', 'center_y']
        for i in range(4): header.extend([f'corner_{i}_x', f'corner_{i}_y'])
        for i in range(9): header.append(f'pose_R_{i//3}{i%3}')
        header.extend(['pose_t_x', 'pose_t_y', 'pose_t_z'])
        csv_writer.writerow(header)
    except IOError as e:
        print(f"Error: Could not open CSV file for writing: {e}")
        sys.exit(1)

    print(f"Processing video: {os.path.basename(video_path)}")
    print(f"Tracking data will be saved to: {output_csv_path}")

    # --- 5. Main Processing Loop ---
    frame_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_counter += 1
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        if frame_counter % 100 == 0:
            print(f"  - {os.path.basename(video_path)}: Processing frame {frame_counter}/{total_frames}")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = at_detector.detect(
            gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=TAG_SIZE_M
        )

        for tag in tags:
            center = tag.center
            corners = tag.corners.flatten()
            pose_R = tag.pose_R.flatten()
            pose_t = tag.pose_t.flatten()
            
            row = [frame_counter, timestamp_ms, tag.tag_id, center[0], center[1]]
            row.extend(corners)
            row.extend(pose_R)
            row.extend(pose_t)
            csv_writer.writerow(row)

    # --- 6. Clean Up ---
    cap.release()
    csv_file.close()
    print(f"Processing complete for: {os.path.basename(video_path)}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        video_file_path = sys.argv[1]
        track_apriltags_in_video(video_file_path)
    else:
        print("No video file provided via command line. Opening file dialog...")
        root = tk.Tk()
        root.withdraw()
        video_file_path = filedialog.askopenfilename(
            title="Select a video file to track",
            initialdir="./Videos",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if video_file_path:
            track_apriltags_in_video(video_file_path)
        else:
            print("No video file selected. Exiting.")
