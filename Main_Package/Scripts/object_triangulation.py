# -*- coding: utf-8 -*-
"""
Calculates the final 3D pose of the CanSat using filled tracking data.

This script takes the complete, gap-filled 3D positions of all tags at every
moment in time and calculates a single, robust 3D pose (position and
orientation) for the CanSat. It then performs a final interpolation to
create a smooth, continuous trajectory.

Standalone Mode:
- Run `python object_triangulation.py`.
- A file dialog will open to select the necessary input files.

Pipeline Mode (called from main.py):
- This script is the final step. It finds its input files automatically.

The script saves the final output to `Outputs/Object_Triangulation/final_cansat_pose.csv`.
"""
import cv2
import numpy as np
import pandas as pd
import os
import sys
import tkinter as tk
from tkinter import filedialog
import csv

def calculate_final_pose(filled_data_path, layout_model_path):
    """
    Main function to run the final pose estimation pipeline.
    """
    # --- 1. Load Input Files ---
    print("Loading filled tracking data and layout model...")
    try:
        filled_df = pd.read_csv(filled_data_path)
        layout_model_df = pd.read_csv(layout_model_path)
        
        tag_layout_model = {
            row['tag_id']: np.array([row['pos_x'], row['pos_y'], row['pos_z']])
            for _, row in layout_model_df.iterrows()
        }
        print("All data loaded successfully.")
    except Exception as e:
        print(f"Error loading input files: {e}")
        sys.exit(1)

    # --- 2. Process Timeline and Calculate Pose for Each Timestamp ---
    print("\nCalculating final pose for each timestamp...")
    output_data = []
    
    grouped_by_time = filled_df.groupby('adjusted_timestamp_ms')

    for timestamp, frame_data in grouped_by_time:
        current_frame_output = {'adjusted_timestamp_ms': timestamp}
        
        local_pts, world_pts = [], []
        
        for _, tag_row in frame_data.iterrows():
            tag_id = int(tag_row['tag_id'])
            if tag_id in tag_layout_model:
                local_pts.append(tag_layout_model[tag_id])
                world_pts.append([tag_row['pos_x'], tag_row['pos_y'], tag_row['pos_z']])

        if len(local_pts) >= 3:
            obj_pts = np.array(local_pts, dtype=np.float32)
            world_pts = np.array(world_pts, dtype=np.float32)
            
            success, transform, inliers = cv2.estimateAffine3D(obj_pts, world_pts)
            
            if success:
                R_cansat, t_cansat = transform[:, :3], transform[:, 3]
                
                # Calculate error based on how many points were inferred
                inferred_count = frame_data['is_inferred'].sum()
                total_count = len(frame_data)
                error_metric = inferred_count / total_count if total_count > 0 else 1.0

                current_frame_output.update({
                    'pos_x': t_cansat[0], 'pos_y': t_cansat[1], 'pos_z': t_cansat[2],
                    **{f'R_{i//3}{i%3}': val for i, val in enumerate(R_cansat.flatten())},
                    'inferred_ratio': error_metric
                })
        
        output_data.append(current_frame_output)

    # --- 3. Interpolate Gaps and Save Final Results ---
    print("\nInterpolating any remaining gaps in the data...")
    pose_columns = ['pos_x', 'pos_y', 'pos_z', 'R_00', 'R_01', 'R_02', 'R_10', 'R_11', 'R_12', 'R_20', 'R_21', 'R_22', 'inferred_ratio']
    output_df = pd.DataFrame(output_data, columns=['adjusted_timestamp_ms'] + pose_columns)
    
    if output_df['pos_x'].isnull().all():
        print("\nWARNING: No valid poses were ever calculated. The output file will be empty.")
        output_df.to_csv('final_cansat_pose.csv', index=False)
        return

    output_df.set_index('adjusted_timestamp_ms', inplace=True)
    output_df.interpolate(method='time', limit_direction='both', inplace=True)

    output_dir = os.path.join("Outputs", "Object_Triangulation")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_filename = os.path.join(output_dir, 'final_cansat_pose.csv')
    output_df.to_csv(output_filename)
    
    print("\nProcessing complete. Final pose data saved to 'final_cansat_pose.csv'")

if __name__ == '__main__':
    print("Running in standalone mode...")
    root = tk.Tk()
    root.withdraw()
    
    print("Select the 'filled_tracking_data.csv' file...")
    filled_data_file = filedialog.askopenfilename(
        title="Select filled_tracking_data.csv",
        initialdir=os.path.join("Outputs", "Object_Triangulation"),
        filetypes=[("CSV File]s", "*.csv")]
    )
    
    print("Select the 'tag_layout_model.csv' file...")
    layout_file = filedialog.askopenfilename(
        title="Select tag_layout_model.csv",
        initialdir=os.path.join("Outputs", "Object_Calibration"),
        filetypes=[("CSV Files", "*.csv")]
    )
    
    if filled_data_file and layout_file:
        calculate_final_pose(filled_data_file, layout_file)
    else:
        print("One or more files were not selected. Exiting.")
