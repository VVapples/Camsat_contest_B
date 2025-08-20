# -*- coding: utf-8 -*-
"""
Verifies video synchronization by playing back multiple video files side-by-side,
using pre-calculated frame offsets from a CSV file.

This script reads 'sync_offsets.csv', prompts the user to select the
corresponding video files, applies the frame delays, and displays a combined
video feed for visual confirmation of the synchronization.
"""
import cv2
import numpy as np
import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog

# --- Configuration ---
# The height to which each video will be resized for consistent display.
DISPLAY_HEIGHT = 480


# --- Main Visualization Logic ---

def verify_sync_offsets():
    """
    Loads sync data and video files to play them back for verification.
    """
    # --- 1. Select and Load the CSV Offsets File ---
    root = tk.Tk()
    root.withdraw()

    print("A file dialog will open. Please select the 'sync_offsets.csv' file.")
    csv_path = filedialog.askopenfilename(
        title="Select sync_offsets.csv",
        filetypes=[("CSV Files", "*.csv")]
    )
    if not csv_path:
        print("No CSV file selected. Exiting.")
        return

    try:
        sync_data = pd.read_csv(csv_path)
        print("Successfully loaded sync data.")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # --- 2. Select the Corresponding Video Files ---
    video_paths = {}
    print("\nNow, please select the video file for each camera listed in the CSV.")
    for index, row in sync_data.iterrows():
        camera_name = row['camera_name']
        
        video_path = filedialog.askopenfilename(
            title=f"Select video file for camera: '{camera_name}'"
        )
        if not video_path:
            print(f"Skipped selecting a video for '{camera_name}'. Exiting.")
            return
        
        video_paths[camera_name] = video_path

    # --- 3. Open Video Captures and Apply Offsets ---
    caps = {}
    for camera_name, path in video_paths.items():
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Error: Could not open video for '{camera_name}'.")
            return
        
        # Get the offset for this camera from the dataframe
        offset = sync_data.loc[sync_data['camera_name'] == camera_name, 'offset_frames'].iloc[0]
        
        # Apply the frame offset
        cap.set(cv2.CAP_PROP_POS_FRAMES, offset)
        caps[camera_name] = cap
        print(f"Opened '{camera_name}' and applied frame offset of {offset}.")

    # --- 4. Main Playback Loop ---
    print("\nStarting synchronized playback... Press 'q' to exit.")
    
    frame_counter = 0
    while True:
        frames = []
        
        # Read one frame from each video
        for camera_name, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                # If any video ends, we stop
                print("End of video reached.")
                break
            
            # --- Resize frame for consistent display ---
            h, w, _ = frame.shape
            scale = DISPLAY_HEIGHT / h
            new_w = int(w * scale)
            resized_frame = cv2.resize(frame, (new_w, DISPLAY_HEIGHT))
            
            # --- Add text overlay ---
            text = f"{camera_name} | Frame: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}"
            cv2.putText(resized_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            frames.append(resized_frame)
        
        if len(frames) != len(caps):
            # This means one of the videos ended
            break

        # Combine frames horizontally
        combined_frame = cv2.hconcat(frames)

        # Display the result
        cv2.imshow("Synchronization Verification", combined_frame)

        if cv2.waitKey(30) & 0xFF == ord('q'): # Adjust waitKey for playback speed
            break
            
        frame_counter += 1

    # --- 5. Clean Up ---
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()
    print("Playback finished.")


if __name__ == '__main__':
    verify_sync_offsets()
