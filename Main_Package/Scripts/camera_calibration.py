# -*- coding: utf-8 -*-
"""
Performs intrinsic camera calibration using a video of a chessboard pattern.

This script is designed to work within the project's file structure.

Standalone Mode:
- Run `python camera_calibration.py`.
- A file dialog will open to select a video.

Pipeline Mode (called from main.py):
- Run `python camera_calibration.py "path/to/video.mp4"`.
- The specified video will be processed directly.

The script saves the output to `Outputs/Calibration/(cameraname).npz`.
"""
import cv2
import numpy as np
import os
import sys
import tkinter as tk
from tkinter import filedialog

# --- Configuration ---
CHESSBOARD_CORNERS_X = 8
CHESSBOARD_CORNERS_Y = 6
SQUARE_SIZE_MM = 25
MIN_VALID_FRAMES = 10
FRAME_INTERVAL_ATTEMPTS = [30, 20, 10, 5]

def calibrate_from_video(video_path):
    """
    Calibrates a camera using the provided video file path, with a retry mechanism.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        sys.exit(1) # Exit with an error code

    # --- Prepare Output Directory and Filename ---
    # Assumes the script is run from the root of Main_Package
    output_dir = os.path.join("Outputs", "Calibration")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
        
    # Get camera name from the parent folder of the video
    folder_path = os.path.dirname(video_path)
    camera_name = os.path.basename(folder_path)
    
    calibration_filename = camera_name + ".npz"
    output_path = os.path.join(output_dir, calibration_filename)

    # Prepare Object Points
    objp = np.zeros((CHESSBOARD_CORNERS_X * CHESSBOARD_CORNERS_Y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_CORNERS_X, 0:CHESSBOARD_CORNERS_Y].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE_MM

    objpoints = []
    imgpoints = []
    gray_shape = None
    
    print(f"Starting calibration for camera: '{camera_name}'")
    print(f"Input video: {os.path.basename(video_path)}")

    # --- Loop through sampling intervals ---
    for interval in FRAME_INTERVAL_ATTEMPTS:
        print(f"\nAttempting to find corners by sampling 1 in every {interval} frames...")
        objpoints.clear()
        imgpoints.clear()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return

        frame_count = 0
        found_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_count += 1
            
            if frame_count % interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if gray_shape is None:
                    gray_shape = gray.shape[::-1]

                ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_CORNERS_X, CHESSBOARD_CORNERS_Y), None)

                if ret:
                    found_count += 1
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                    imgpoints.append(corners2)
        
        cap.release()
        
        print(f"  - Found corners in {found_count} frames.")
        
        if len(objpoints) >= MIN_VALID_FRAMES:
            print("Sufficient number of views found.")
            break
        else:
            print(f"  - Not enough views found ({len(objpoints)} < {MIN_VALID_FRAMES}). Trying a more frequent sample rate...")

    # --- Perform Calibration ---
    if len(objpoints) < MIN_VALID_FRAMES:
        print(f"\nFATAL: Calibration failed for '{camera_name}'.")
        print("Could not find enough valid views even with the most frequent sampling.")
        sys.exit(1) # Exit with an error code

    print("\nPerforming calibration...")
    try:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)
        
        if ret:
            print("\nCalibration successful!")
            np.savez(output_path, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
            print(f"Calibration data saved to '{output_path}'")
        else:
            print("\nFATAL: Calibration failed during calculation.")
            sys.exit(1)

    except Exception as e:
        print(f"\nAn error occurred during calibration: {e}")
        sys.exit(1)


if __name__ == '__main__':
    # Check if a file path was provided as a command-line argument
    if len(sys.argv) > 1:
        video_file_path = sys.argv[1]
        calibrate_from_video(video_file_path)
    else:
        # If no argument, fall back to the file dialog
        print("No video file provided via command line. Opening file dialog...")
        root = tk.Tk()
        root.withdraw()
        video_file_path = filedialog.askopenfilename(
            title="Select a calibration video file",
            initialdir="./Videos", # Start in the Videos folder
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if video_file_path:
            calibrate_from_video(video_file_path)
        else:
            print("No video file selected. Exiting.")
