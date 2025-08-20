# -*- coding: utf-8 -*-
"""
Performs 3D triangulation and pose estimation of an object using synchronized,
timestamp-aware AprilTag tracking data from multiple cameras.
"""
import cv2
import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
import glob
import itertools
import csv

# --- Configuration ---
CANSAT_TAG_LAYOUT = {
    1: np.array([0.05, 0, 0]),
    2: np.array([0, 0.05, 0]),
    3: np.array([0, 0, 0.05]),
}
TIMESTAMP_TOLERANCE_MS = 10

def triangulate_and_estimate_pose():
    """Main function to run the entire triangulation and pose estimation pipeline."""
    root = tk.Tk()
    root.withdraw()

    print("Loading input files...")
    print("\nA file dialog will open. Please select your final, adjusted synchronization file (e.g., 'sync_offsets_final.csv').")
    sync_csv_path = filedialog.askopenfilename(
        title="Select the 'sync_offsets_final.csv' file",
        filetypes=[("CSV Files", "*.csv"), ("All files", "*.*")]
    )
    if not sync_csv_path: return
    sync_df = pd.read_csv(sync_csv_path)

    calib_files = glob.glob("calibration/*.npz")
    camera_params = {}
    for f in calib_files:
        cam_name = os.path.splitext(os.path.basename(f))[0]
        data = np.load(f)
        camera_params[cam_name] = {'mtx': data['mtx'], 'dist': data['dist']}
    
    tracker_files = glob.glob("trackerdata/**/*.csv", recursive=True)
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
        print("Error: No valid tracker data files found.")
        return
        
    master_df = pd.concat(all_tracker_data, ignore_index=True)
    print("All input files loaded successfully.")

    print("\nAdjusting timestamps to a common timeline...")
    master_df['adjusted_timestamp_ms'] = 0.0
    for index, row in sync_df.iterrows():
        cam_name = row['camera_name']
        sync_time = row['sync_timestamp_ms']
        master_df.loc[master_df['camera'] == cam_name, 'adjusted_timestamp_ms'] = master_df['timestamp_ms'] - sync_time
    
    master_df = master_df[master_df['adjusted_timestamp_ms'] >= 0].sort_values('adjusted_timestamp_ms')

    print("\nCalculating relative camera poses...")
    camera_poses = {}
    reference_cam_name = sync_df.sort_values('sync_timestamp_ms').iloc[0]['camera_name']
    camera_poses[reference_cam_name] = {'R': np.eye(3), 't': np.zeros((3, 1))}

    # FIX: Iterate through each camera and merge it with the reference camera individually.
    # This creates predictable column names and is more robust.
    ref_cam_data = master_df[master_df['camera'] == reference_cam_name].copy()
    for cam_name in camera_params:
        if cam_name == reference_cam_name: continue

        other_cam_data = master_df[master_df['camera'] == cam_name].copy()
        
        merged = pd.merge_asof(
            ref_cam_data.sort_values('adjusted_timestamp_ms'),
            other_cam_data.sort_values('adjusted_timestamp_ms'),
            on='adjusted_timestamp_ms',
            by='tag_id',
            direction='nearest',
            tolerance=TIMESTAMP_TOLERANCE_MS,
            suffixes=('_ref', '_other')
        ).dropna() # Drop rows where no match was found

        if merged.empty:
            print(f"Warning: No common tags found between '{reference_cam_name}' and '{cam_name}'. Cannot determine pose.")
            continue

        common_detection = merged.iloc[0]
        
        R_ref = common_detection[[f'pose_R_{i//3}{i%3}_ref' for i in range(9)]].values.astype(np.float64).reshape(3, 3)
        t_ref = common_detection[['pose_t_x_ref', 'pose_t_y_ref', 'pose_t_z_ref']].values.astype(np.float64).reshape(3, 1)

        R_other = common_detection[[f'pose_R_{i//3}{i%3}_other' for i in range(9)]].values.astype(np.float64).reshape(3, 3)
        t_other = common_detection[['pose_t_x_other', 'pose_t_y_other', 'pose_t_z_other']].values.astype(np.float64).reshape(3, 1)
        
        R_other_to_ref = R_ref @ R_other.T
        t_other_to_ref = t_ref - (R_ref @ R_other.T @ t_other)
        
        camera_poses[cam_name] = {'R': R_other_to_ref, 't': t_other_to_ref}
        print(f"  - Pose calculated for camera '{cam_name}'")

    with open('camera_poses.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['camera_name', 'R', 't'])
        for name, pose in camera_poses.items():
            writer.writerow([name, pose['R'].flatten().tolist(), pose['t'].flatten().tolist()])
    print("Camera poses saved to 'camera_poses.csv'")

    print("\nStarting triangulation and pose estimation...")
    world_origin = None
    output_data = []

    # Group by time bins to process chunks of data together
    time_bins = np.arange(0, master_df["adjusted_timestamp_ms"].max() + 50, 50)
    grouped_by_time = master_df.groupby(pd.cut(master_df["adjusted_timestamp_ms"], time_bins, right=False))

    for time_bin, frame_data in grouped_by_time:
        if frame_data.empty: continue
        
        avg_timestamp = frame_data['adjusted_timestamp_ms'].mean()
        current_frame_output = {'adjusted_timestamp_ms': avg_timestamp}
        visible_tags = frame_data.groupby('tag_id')
        triangulated_points = {}

        for tag_id, tag_detections in visible_tags:
            if len(tag_detections) < 2: continue

            for cam1_data, cam2_data in itertools.combinations(tag_detections.iterrows(), 2):
                _, cam1 = cam1_data
                _, cam2 = cam2_data
                
                cam1_name, cam2_name = cam1['camera'], cam2['camera']
                
                # Check if poses for both cameras were successfully calculated
                if cam1_name not in camera_poses or cam2_name not in camera_poses:
                    continue

                K1, K2 = camera_params[cam1_name]['mtx'], camera_params[cam2_name]['mtx']
                R1, t1 = camera_poses[cam1_name]['R'], camera_poses[cam1_name]['t']
                R2, t2 = camera_poses[cam2_name]['R'], camera_poses[cam2_name]['t']
                
                P1 = K1 @ np.hstack((R1, t1))
                P2 = K2 @ np.hstack((R2, t2))
                
                pt1 = np.array([cam1['center_x'], cam1['center_y']])
                pt2 = np.array([cam2['center_x'], cam2['center_y']])
                
                point_4d_hom = cv2.triangulatePoints(P1, P2, pt1.reshape(2, 1), pt2.reshape(2, 1))
                point_3d = (point_4d_hom / point_4d_hom[3])[:3].flatten()
                
                if world_origin is None:
                    world_origin = point_3d
                
                triangulated_points[tag_id] = point_3d - world_origin
                break 

        if len(triangulated_points) >= 3:
            object_points_local, world_points_3d = [], []
            
            for tag_id, point_3d in triangulated_points.items():
                if tag_id in CANSAT_TAG_LAYOUT:
                    object_points_local.append(CANSAT_TAG_LAYOUT[tag_id])
                    world_points_3d.append(point_3d)
            
            if len(object_points_local) >= 3:
                obj_pts = np.array(object_points_local, dtype=np.float32)
                world_pts = np.array(world_points_3d, dtype=np.float32)
                
                success, transform, inliers = cv2.estimateAffine3D(obj_pts, world_pts)
                
                if success:
                    R_cansat, t_cansat = transform[:, :3], transform[:, 3]
                    error_radius = 0.0
                    if inliers is not None:
                        outliers = np.where(inliers == 0)[0]
                        if len(outliers) > 0:
                            projected_pts = (R_cansat @ obj_pts.T).T + t_cansat
                            errors = np.linalg.norm(projected_pts[outliers] - world_pts[outliers], axis=1)
                            error_radius = np.mean(errors)
                    
                    current_frame_output.update({
                        'pos_x': t_cansat[0], 'pos_y': t_cansat[1], 'pos_z': t_cansat[2],
                        **{f'R_{i//3}{i%3}': val for i, val in enumerate(R_cansat.flatten())},
                        'error_radius': error_radius
                    })
        
        output_data.append(current_frame_output)

    print("\nInterpolating missing data points...")
    pose_columns = ['pos_x', 'pos_y', 'pos_z', 'R_00', 'R_01', 'R_02', 'R_10', 'R_11', 'R_12', 'R_20', 'R_21', 'R_22', 'error_radius']
    output_df = pd.DataFrame(output_data, columns=['adjusted_timestamp_ms'] + pose_columns)
    
    if output_df['pos_x'].isnull().all():
        print("\nWARNING: No valid poses were ever calculated.")
        output_df.to_csv('triangulated_pose_data.csv', index=False)
        return

    output_df.set_index('adjusted_timestamp_ms', inplace=True)
    output_df.interpolate(method='time', limit_direction='both', inplace=True)
    output_df.to_csv('triangulated_pose_data.csv')
    print("\nProcessing complete. Final pose data saved to 'triangulated_pose_data.csv'")

if __name__ == '__main__':
    triangulate_and_estimate_pose()
