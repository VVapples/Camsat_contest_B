# -*- coding: utf-8 -*-
"""
Visualizes pre-recorded AprilTag tracking data onto its source video.

This script reads a video file and a corresponding CSV data file (generated
by the tracking script), then creates a new video with the tracking
visualizations (bounding boxes, pose axes) drawn on each frame.
"""
import cv2
import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog

# --- Configuration ---
# This must match the tag size used during the initial tracking.
TAG_SIZE_M = 0.055 


# --- Main Visualization Logic ---

def visualize_tracking_data():
    """
    Main function to run the visualization process.
    """
    # --- 1. File Selection Dialogs ---
    root = tk.Tk()
    root.withdraw() # Hide the main tkinter window

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Select Video File
    print("A file dialog will open. Please select the ORIGINAL video file.")
    video_path = filedialog.askopenfilename(
        title="Select the ORIGINAL video file",
        initialdir=script_dir,
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )
    if not video_path:
        print("No video file selected. Exiting.")
        return

    # Select CSV Data File
    print("A file dialog will open. Please select the corresponding CSV tracking data file.")
    csv_path = filedialog.askopenfilename(
        title="Select the corresponding CSV tracking data file",
        initialdir=os.path.join(script_dir, "trackerdata"),
        filetypes=[("CSV Files", "*.csv"), ("All files", "*.*")]
    )
    if not csv_path:
        print("No CSV file selected. Exiting.")
        return

    # --- 2. Automatically Find and Load Calibration Data ---
    try:
        # Get the folder name from the video path to find the calibration file
        folder_path = os.path.dirname(video_path)
        folder_name = os.path.basename(folder_path)
        
        calibration_filename = folder_name + ".npz"
        calibration_path = os.path.join("calibration", calibration_filename)

        if not os.path.exists(calibration_path):
            print(f"Error: Calibration file not found at expected path: {calibration_path}")
            return

        with np.load(calibration_path) as data:
            mtx = data['mtx']
            dist = data['dist']
        print(f"Successfully loaded calibration data from {calibration_path}")

    except Exception as e:
        print(f"Error loading calibration file: {e}")
        return

    # --- 3. Load and Process Tracking Data using Pandas ---
    try:
        print("Loading tracking data from CSV...")
        df = pd.read_csv(csv_path)
        # Group data by frame number for quick lookup
        grouped_data = df.groupby('frame')
        print("Tracking data loaded successfully.")
    except Exception as e:
        print(f"Error reading or processing CSV file: {e}")
        return

    # --- 4. Prepare Video Input and Output ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir = "visualized_videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_filename = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = os.path.join(output_dir, f"{base_filename}_visualized.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    print(f"Visualizing data... Press 'q' in the display window to stop early.")
    print(f"Output will be saved to: {output_filename}")

    # --- 5. Main Processing Loop ---
    frame_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_counter += 1
        
        # Check if there is tracking data for the current frame
        if frame_counter in grouped_data.groups:
            frame_detections = grouped_data.get_group(frame_counter)
            
            for index, tag in frame_detections.iterrows():
                # Reconstruct data from the CSV row
                tag_id = int(tag['tag_id'])
                center = np.array([tag['center_x'], tag['center_y']])
                
                corners = np.array([
                    [tag['corner_0_x'], tag['corner_0_y']],
                    [tag['corner_1_x'], tag['corner_1_y']],
                    [tag['corner_2_x'], tag['corner_2_y']],
                    [tag['corner_3_x'], tag['corner_3_y']]
                ]).astype(int)
                
                pose_R = np.array([
                    [tag['pose_R_00'], tag['pose_R_01'], tag['pose_R_02']],
                    [tag['pose_R_10'], tag['pose_R_11'], tag['pose_R_12']],
                    [tag['pose_R_20'], tag['pose_R_21'], tag['pose_R_22']]
                ])
                
                pose_t = np.array([tag['pose_t_x'], tag['pose_t_y'], tag['pose_t_z']])

                # --- Draw visualizations on frame ---
                cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
                
                center_int = center.astype(int)
                cv2.putText(frame, str(tag_id), (center_int[0], center_int[1] - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.drawFrameAxes(frame, mtx, dist, pose_R, pose_t, length=TAG_SIZE_M * 0.8)

        # Write the processed frame to the output file
        out.write(frame)

        # Display the resulting frame
        cv2.imshow('AprilTag Visualization', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 6. Clean Up ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Visualization complete.")

if __name__ == '__main__':
    visualize_tracking_data()
