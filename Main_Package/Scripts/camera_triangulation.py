# -*- coding: utf-8 -*-
"""
Calculates the relative 3D poses (position and orientation) of multiple
cameras based on shared AprilTag detections in synchronized video streams.

Standalone Mode:
- Run `python camera_triangulation.py`.
- A file dialog will open to select the sync_offsets_final.csv file.

Pipeline Mode (called from main.py):
- This script is typically run after syncing and tracking. It does not
  require command-line arguments as it finds its input files based on
  the project structure.

The script saves the output to `Outputs/Camera_Triangulation/camera_poses.csv`.
"""
import cv2
import numpy as np
import pandas as pd
import os
import sys
import tkinter as tk
from tkinter import filedialog
import glob
import itertools
import csv

# --- Configuration ---
# Tolerance for matching timestamps between cameras, in milliseconds.
TIMESTAMP_TOLERANCE_MS = 15

def calculate_camera_poses(sync_csv_path):
    """
    Main function to run the camera pose calculation pipeline.
    """
    if not os.path.exists(sync_csv_path):
        print(f"Error: Sync file not found at '{sync_csv_path}'")
        sys.exit(1)

    # --- 1. Load Sync and Tracker Data ---
    print("Loading sync and tracker data...")
    sync_df = pd.read_csv(sync_csv_path)
    
    tracker_files = glob.glob(os.path.join("Outputs", "Tracking", "**", "*.csv"), recursive=True)
    all_tracker_data = []
    for f in tracker_files:
        cam_name = os.path.basename(os.path.dirname(f))
        try:
            df = pd.read_csv(f)
            df['camera'] = cam_name
            all_tracker_data.append(df)
        except pd.errors.EmptyDataError:
            print(f"Warning: Tracker file is empty, skipping: {f}")
    
    if not all_tracker_data:
        print("Error: No valid tracker data files found in 'Outputs/Tracking/'.")
        sys.exit(1)
        
    master_df = pd.concat(all_tracker_data, ignore_index=True)
    print("All input files loaded successfully.")

    # --- 2. Adjust Timestamps to a Common Timeline ---
    print("\nAdjusting timestamps to a common timeline...")
    master_df['adjusted_timestamp_ms'] = 0.0
    for index, row in sync_df.iterrows():
        cam_name = row['camera_name']
        sync_time = row['sync_timestamp_ms']
        master_df.loc[master_df['camera'] == cam_name, 'adjusted_timestamp_ms'] = master_df['timestamp_ms'] - sync_time
    
    master_df = master_df[master_df['adjusted_timestamp_ms'] >= 0].sort_values('adjusted_timestamp_ms')

    # --- 3. Calculate Relative Camera Poses ---
    print("\nCalculating relative camera poses...")
    camera_poses = {}
    
    # Determine the reference camera (the one with the earliest sync time)
    reference_cam_name = sync_df.sort_values('sync_timestamp_ms').iloc[0]['camera_name']
    camera_poses[reference_cam_name] = {'R': np.eye(3), 't': np.zeros((3, 1))}
    print(f"Reference camera set to: '{reference_cam_name}'")

    ref_cam_data = master_df[master_df['camera'] == reference_cam_name].copy()
    
    camera_names = [name for name in sync_df['camera_name'].unique() if name != reference_cam_name]

    for cam_name in camera_names:
        other_cam_data = master_df[master_df['camera'] == cam_name].copy()
        
        # Find moments where both cameras saw the same tag at roughly the same time
        merged = pd.merge_asof(
            ref_cam_data.sort_values('adjusted_timestamp_ms'),
            other_cam_data.sort_values('adjusted_timestamp_ms'),
            on='adjusted_timestamp_ms',
            by='tag_id',
            direction='nearest',
            tolerance=TIMESTAMP_TOLERANCE_MS,
            suffixes=('_ref', '_other')
        ).dropna()

        if merged.empty:
            print(f"Warning: No common tags found between '{reference_cam_name}' and '{cam_name}'. Cannot determine pose.")
            continue

        # Use the first common detection to find the transformation
        common_detection = merged.iloc[0]
        
        try:
            R_ref = common_detection[[f'pose_R_{i//3}{i%3}_ref' for i in range(9)]].values.astype(np.float64).reshape(3, 3)
            t_ref = common_detection[['pose_t_x_ref', 'pose_t_y_ref', 'pose_t_z_ref']].values.astype(np.float64).reshape(3, 1)

            R_other = common_detection[[f'pose_R_{i//3}{i%3}_other' for i in range(9)]].values.astype(np.float64).reshape(3, 3)
            t_other = common_detection[['pose_t_x_other', 'pose_t_y_other', 'pose_t_z_other']].values.astype(np.float64).reshape(3, 1)
            
            # Math to find pose of other_cam relative to ref_cam
            R_other_to_ref = R_ref @ R_other.T
            t_other_to_ref = t_ref - (R_ref @ R_other.T @ t_other)
            
            camera_poses[cam_name] = {'R': R_other_to_ref, 't': t_other_to_ref}
            print(f"  - Pose calculated for camera '{cam_name}'")
        except Exception as e:
            print(f"An error occurred processing poses for '{cam_name}': {e}")


    # --- 4. Save Camera Poses ---
    output_dir = os.path.join("Outputs", "Camera_Triangulation")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_filename = os.path.join(output_dir, 'camera_poses.csv')
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['camera_name', 'R', 't'])
        for name, pose in camera_poses.items():
            writer.writerow([name, list(pose['R'].flatten()), list(pose['t'].flatten())])
    
    print(f"\nCamera poses saved to '{output_filename}'")
    print("Camera triangulation complete.")


if __name__ == '__main__':
    # This script is not designed to be run with command-line arguments,
    # as it finds its inputs automatically. The main.py script will just call it.
    # Standalone mode uses a file dialog.
    
    print("Running in standalone mode...")
    root = tk.Tk()
    root.withdraw()
    
    sync_file_path = filedialog.askopenfilename(
        title="Select the 'sync_offsets_final.csv' file",
        initialdir=os.path.join("Outputs", "Syncing"),
        filetypes=[("CSV Files", "*.csv")]
    )
    
    if sync_file_path:
        calculate_camera_poses(sync_file_path)
    else:
        print("No sync file selected. Exiting.")
