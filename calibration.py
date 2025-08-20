# -*- coding: utf-8 -*-
"""
Performs intrinsic camera calibration using a video file of a chessboard pattern.

This script can be run in two ways:
1. Standalone: `python calibration.py`
   - It will open a file dialog to select a video.
2. With Command-Line Argument: `python calibration.py "path/to/video.mp4"`
   - It will process the video file specified in the argument.

If the script fails to find enough valid frames, it will automatically retry
with a more frequent frame sampling rate.
"""
import cv2
import numpy as np
import os
import sys
import tkinter as tk
from tkinter import filedialog

# --- Configuration ---
CHESSBOARD_CORNERS_X = 8 # Number of inner corners on the X-axis
CHESSBOARD_CORNERS_Y = 6 # Number of inner corners on the Y-axis
SQUARE_SIZE_MM = 25    # The side length of a square in your real-world pattern
MIN_VALID_FRAMES = 10  # The minimum number of views needed for a good calibration.

# --- Frame Sampling Strategy ---
# The script will try these intervals in order until it finds enough frames.
FRAME_INTERVAL_ATTEMPTS = [30, 20, 10, 5]


def calibrate_from_video(video_path):
    """
    Calibrates a camera using the provided video file path, with a retry mechanism.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return

    # --- Prepare Output Directory and Filename ---
    output_dir = "calibration"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
        
    folder_path = os.path.dirname(video_path)
    folder_name = os.path.basename(folder_path)
    
    calibration_filename = folder_name + ".npz"
    output_path = os.path.join(output_dir, calibration_filename)

    # Prepare Object Points (3D points in real-world space)
    objp = np.zeros((CHESSBOARD_CORNERS_X * CHESSBOARD_CORNERS_Y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_CORNERS_X, 0:CHESSBOARD_CORNERS_Y].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE_MM

    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane.
    gray_shape = None
    
    print(f"Starting video processing for: {os.path.basename(video_path)}")
    print(f"Output will be saved as: {output_path}")

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
            if not ret:
                break

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
        
        # If we found enough frames, we can stop trying and proceed
        if len(objpoints) >= MIN_VALID_FRAMES:
            print("Sufficient number of views found.")
            break
        else:
            print(f"  - Not enough views found ({len(objpoints)} < {MIN_VALID_FRAMES}). Trying a more frequent sample rate...")

    # --- Perform Calibration ---
    if len(objpoints) < MIN_VALID_FRAMES:
        print(f"\nCalibration failed for '{os.path.basename(video_path)}'.")
        print("Could not find enough valid views even with the most frequent sampling.")
        return

    print("\nPerforming calibration...")
    try:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)
        
        if ret:
            print("\nCalibration successful!")
            np.savez(output_path, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
            print(f"Calibration data saved to '{output_path}'")
        else:
            print("\nCalibration failed.")

    except Exception as e:
        print(f"\nAn error occurred during calibration: {e}")


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
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if video_file_path:
            calibrate_from_video(video_file_path)
        else:
            print("No video file selected. Exiting.")
