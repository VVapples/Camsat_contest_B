# -*- coding: utf-8 -*-
"""
Fills in missing AprilTag detections using a known 3D model of the tag layout.

This script implements model-based tracking. For each moment in time, it
triangulates the positions of all visible tags. If not all tags are visible,
it uses the known layout model and the pose from the last good frame to
infer the 3D positions of the missing tags.

Standalone Mode:
- Run `python tag_filler.py`.
- A file dialog will open to select the necessary input files.

Pipeline Mode (called from main.py):
- This script is typically run after camera triangulation. It finds its
  input files automatically based on the project structure.

The script saves the output to `Outputs/Object_Triangulation/filled_tracking_data.csv`.
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
TIMESTAMP_TOLERANCE_MS = 15

def fill_missing_tags(sync_path, camera_poses_path, layout_model_path):
    """
    Main function to run the tag filling pipeline.
    """
    # --- 1. Load All Input Files ---
    print("Loading all necessary data files...")
    try:
        sync_df = pd.read_csv(sync_path)
        camera_poses_df = pd.read_csv(camera_poses_path, converters={'R': pd.eval, 't': pd.eval})
        layout_model_df = pd.read_csv(layout_model_path)
        
        camera_poses = {row['camera_name']: {'R': np.array(row['R']).reshape(3,3), 't': np.array(row['t']).reshape(3,1)} for _, row in camera_poses_df.iterrows()}
        tag_layout_model = {row['tag_id']: row.to_dict() for _, row in layout_model_df.iterrows()}

        tracker_files = glob.glob(os.path.join("Outputs", "Tracking", "**", "*.csv"), recursive=True)
        master_df = pd.concat([pd.read_csv(f).assign(camera=os.path.basename(os.path.dirname(f))) for f in tracker_files])
        
        print("All data loaded successfully.")
    except Exception as e:
        print(f"Error loading input files: {e}")
        sys.exit(1)

    # --- 2. Adjust Timestamps ---
    print("\nAdjusting timestamps to a common timeline...")
    master_df['adjusted_timestamp_ms'] = 0.0
    for _, row in sync_df.iterrows():
        master_df.loc[master_df['camera'] == row['camera_name'], 'adjusted_timestamp_ms'] = master_df['timestamp_ms'] - row['sync_timestamp_ms']
    master_df = master_df[master_df['adjusted_timestamp_ms'] >= 0].sort_values('adjusted_timestamp_ms')

    # --- 3. Triangulate and Fill Gaps ---
    print("\nProcessing timeline to triangulate and fill missing tags...")
    world_origin = None
    last_good_pose = {'R': np.eye(3), 't': np.zeros(3)} # Keep track of the last known good pose
    all_filled_points = []

    time_bins = np.arange(0, master_df["adjusted_timestamp_ms"].max() + 50, 50)
    grouped_by_time = master_df.groupby(pd.cut(master_df["adjusted_timestamp_ms"], time_bins, right=False))

    for time_bin, frame_data in grouped_by_time:
        if frame_data.empty: continue
        
        avg_timestamp = frame_data['adjusted_timestamp_ms'].mean()
        triangulated_points = {} # {tag_id: 3d_point}

        # --- Triangulate all visible tags in this time bin ---
        for tag_id, detections in frame_data.groupby('tag_id'):
            if len(detections) < 2: continue
            
            for cam1_data, cam2_data in itertools.combinations(detections.iterrows(), 2):
                _, cam1 = cam1_data
                _, cam2 = cam2_data
                cam1_name, cam2_name = cam1['camera'], cam2['camera']

                if cam1_name not in camera_poses or cam2_name not in camera_poses: continue

                K1, K2 = camera_params[cam1_name]['mtx'], camera_params[cam2_name]['mtx']
                R1, t1 = camera_poses[cam1_name]['R'], camera_poses[cam1_name]['t']
                R2, t2 = camera_poses[cam2_name]['R'], camera_poses[cam2_name]['t']
                
                P1, P2 = K1 @ np.hstack((R1, t1)), K2 @ np.hstack((R2, t2))
                pt1, pt2 = np.array([cam1['center_x'], cam1['center_y']]), np.array([cam2['center_x'], cam2['center_y']])
                
                point_4d = cv2.triangulatePoints(P1, P2, pt1.reshape(2, 1), pt2.reshape(2, 1))
                point_3d = (point_4d / point_4d[3])[:3].flatten()
                
                if world_origin is None: world_origin = point_3d
                
                triangulated_points[tag_id] = point_3d - world_origin
                break
        
        # --- Determine current pose and fill missing tags ---
        local_pts, world_pts = [], []
        for tag_id, point_3d in triangulated_points.items():
            if tag_id in tag_layout_model:
                local_pts.append([tag_layout_model[tag_id]['pos_x'], tag_layout_model[tag_id]['pos_y'], tag_layout_model[tag_id]['pos_z']])
                world_pts.append(point_3d)

        current_pose = None
        if len(local_pts) >= 3:
            success, transform, _ = cv2.estimateAffine3D(np.array(local_pts), np.array(world_pts))
            if success:
                current_pose = {'R': transform[:,:3], 't': transform[:,3]}
                last_good_pose = current_pose # Update the last known good pose
        
        # If we couldn't get a new pose, use the last good one as a guess
        if current_pose is None:
            current_pose = last_good_pose

        # --- Generate filled data for this timestamp ---
        for tag_id, model_pose in tag_layout_model.items():
            is_inferred = 1
            if tag_id in triangulated_points:
                # This point was actually measured
                final_point = triangulated_points[tag_id]
                is_inferred = 0
            else:
                # This point is a "ghost" - infer its position from the model
                local_pos = np.array([model_pose['pos_x'], model_pose['pos_y'], model_pose['pos_z']])
                final_point = (current_pose['R'] @ local_pos) + current_pose['t']

            all_filled_points.append([avg_timestamp, tag_id, *final_point, is_inferred])

    # --- 4. Save the Filled Data ---
    output_dir = os.path.join("Outputs", "Object_Triangulation")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_filename = os.path.join(output_dir, 'filled_tracking_data.csv')
    header = ['adjusted_timestamp_ms', 'tag_id', 'pos_x', 'pos_y', 'pos_z', 'is_inferred']
    
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_filled_points)
        
    print(f"\nGap-filling complete. Data saved to '{output_filename}'")


if __name__ == '__main__':
    print("Running in standalone mode...")
    root = tk.Tk()
    root.withdraw()
    
    print("Select the 'sync_offsets_final.csv' file...")
    sync_file = filedialog.askopenfilename(title="Select sync_offsets_final.csv", initialdir=os.path.join("Outputs", "Syncing"))
    
    print("Select the 'camera_poses.csv' file...")
    poses_file = filedialog.askopenfilename(title="Select camera_poses.csv", initialdir=os.path.join("Outputs", "Camera_Triangulation"))
    
    print("Select the 'tag_layout_model.csv' file...")
    layout_file = filedialog.askopenfilename(title="Select tag_layout_model.csv", initialdir=os.path.join("Outputs", "Object_Calibration"))
    
    if sync_file and poses_file and layout_file:
        # HACK: Need to load camera params for the triangulation step inside the loop
        # This is a simplification for the standalone script.
        calib_files = glob.glob(os.path.join("Outputs", "Calibration", "*.npz"))
        camera_params = {}
        for f in calib_files:
            cam_name = os.path.splitext(os.path.basename(f))[0]
            data = np.load(f)
            camera_params[cam_name] = {'mtx': data['mtx'], 'dist': data['dist']}
        
        fill_missing_tags(sync_file, poses_file, layout_file)
    else:
        print("One or more files were not selected. Exiting.")
