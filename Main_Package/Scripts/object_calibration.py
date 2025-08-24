# -*- coding: utf-8 -*-
"""
Analyzes a video to automatically determine the relative 3D positions and
orientations of multiple AprilTags, creating a reusable model of the tag layout.

Standalone Mode:
- Run `python object_calibration.py`.
- A file dialog will open to select a video.

Pipeline Mode (called from main.py):
- Run `python object_calibration.py "path/to/video.mp4"`.
- The specified video will be processed directly.

The script saves the output to `Outputs/Object_Calibration/tag_layout_model.csv`.
"""
import cv2
import numpy as np
import pandas as pd
import os
import sys
import tkinter as tk
from tkinter import filedialog
from pupil_apriltags import Detector
import csv

# --- Configuration ---
TAG_FAMILY = 'tag36h11'
TAG_SIZE_M = 0.055 # The size of your AprilTag in meters.
REFERENCE_TAG_ID = 0 # The script will measure all other tags relative to this one.

def generate_tag_layout_model(video_path):
    """
    Main function to run the tag layout generation process.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        sys.exit(1)

    # --- 1. Load Calibration Data ---
    try:
        # Get camera name from the parent folder of the video
        folder_path = os.path.dirname(video_path)
        camera_name = os.path.basename(folder_path)
        calibration_path = os.path.join("Outputs", "Calibration", f"{camera_name}.npz")
        
        with np.load(calibration_path) as data:
            mtx, dist = data['mtx'], data['dist']
        print(f"Successfully loaded calibration data from {calibration_path}")
    except Exception as e:
        print(f"Error: Could not load calibration file for camera '{camera_name}'.")
        print(f"Please ensure '{calibration_path}' exists.")
        print(f"Details: {e}")
        sys.exit(1)

    # --- 2. Initialize Detector and Process Video ---
    camera_params = [mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2]]
    at_detector = Detector(families=TAG_FAMILY, nthreads=4)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        sys.exit(1)

    all_detections = []
    frame_count = 0
    print("Processing video to find tag poses...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = at_detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=TAG_SIZE_M)
        
        for tag in tags:
            all_detections.append({
                'frame': frame_count,
                'tag_id': tag.tag_id,
                'R': tag.pose_R,
                't': tag.pose_t.flatten()
            })

    cap.release()
    if not all_detections:
        print("No tags were detected in the video. Exiting.")
        sys.exit(1)
    
    print(f"Processing complete. Found {len(all_detections)} total tag detections.")

    # --- 3. Calculate Relative Poses ---
    df = pd.DataFrame(all_detections)
    ref_frames = df[df['tag_id'] == REFERENCE_TAG_ID]
    if ref_frames.empty:
        print(f"Error: Reference tag (ID {REFERENCE_TAG_ID}) was not found in the video.")
        sys.exit(1)

    relative_poses = []
    for index, ref_row in ref_frames.iterrows():
        ref_R, ref_t = ref_row['R'], ref_row['t']
        frame_num = ref_row['frame']
        
        other_tags_in_frame = df[(df['frame'] == frame_num) & (df['tag_id'] != REFERENCE_TAG_ID)]
        
        for _, other_row in other_tags_in_frame.iterrows():
            other_R, other_t = other_row['R'], other_row['t']
            
            relative_t = ref_R.T @ (other_t - ref_t)
            relative_R = ref_R.T @ other_R
            
            pose_data = {
                'tag_id': other_row['tag_id'],
                'pos_x': relative_t[0], 'pos_y': relative_t[1], 'pos_z': relative_t[2],
            }
            pose_data.update({f'R_{i//3}{i%3}': val for i, val in enumerate(relative_R.flatten())})
            relative_poses.append(pose_data)

    if not relative_poses:
        print("Could not find any frames with the reference tag and other tags visible at the same time.")
        sys.exit(1)

    # --- 4. Average the Results and Save ---
    rel_pose_df = pd.DataFrame(relative_poses)
    final_layout = rel_pose_df.groupby('tag_id').mean().reset_index()

    for index, row in final_layout.iterrows():
        avg_R = row[[f'R_{i//3}{i%3}' for i in range(9)]].values.reshape(3, 3)
        U, S, Vt = np.linalg.svd(avg_R)
        corrected_R = U @ Vt
        for i, val in enumerate(corrected_R.flatten()):
            final_layout.at[index, f'R_{i//3}{i%3}'] = val

    ref_tag_data = {
        'tag_id': REFERENCE_TAG_ID, 'pos_x': 0, 'pos_y': 0, 'pos_z': 0,
        **{f'R_{i//3}{i%3}': val for i, val in enumerate(np.eye(3).flatten())}
    }
    final_layout = pd.concat([pd.DataFrame([ref_tag_data]), final_layout], ignore_index=True).sort_values('tag_id')

    output_dir = os.path.join("Outputs", "Object_Calibration")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_filename = os.path.join(output_dir, 'tag_layout_model.csv')
    final_layout.to_csv(output_filename, index=False)
    
    print("\n--- Tag Layout Model Generation Complete ---")
    print(f"Model saved to '{output_filename}'")
    print("\nFinal Layout (relative to Tag 0):")
    print(final_layout.to_string())

if __name__ == '__main__':
    if len(sys.argv) > 1:
        video_file_path = sys.argv[1]
        generate_tag_layout_model(video_file_path)
    else:
        print("No video file provided via command line. Opening file dialog...")
        root = tk.Tk()
        root.withdraw()
        video_file_path = filedialog.askopenfilename(
            title="Select a video for layout generation",
            initialdir="./Videos",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if video_file_path:
            generate_tag_layout_model(video_file_path)
        else:
            print("No video file selected. Exiting.")
