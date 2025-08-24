# -*- coding: utf-8 -*-
"""
Finds, fine-tunes, and verifies video synchronization using TIMESTAMPS.

This script is designed to be run as an interactive step in the pipeline.

It first automatically analyzes multiple videos to find an approximate
sync point based on a bright flash. It then allows the user to manually
adjust the sync frame for each video. Finally, it saves the new, accurate
offsets (as timestamps) to `Outputs/Syncing/sync_offsets_final.csv` and
displays a side-by-side image of the synchronized frames for verification.
"""
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import csv

# --- Configuration ---
DISPLAY_HEIGHT = 480
SEARCH_DURATION_SECONDS = 10

def manually_select_sync_frame(video_path, initial_frame):
    """
    Opens a video at a specific frame and allows the user to navigate
    frame-by-frame to select the precise sync frame.
    
    Returns a tuple: (selected_frame_number, timestamp_in_ms, selected_frame_image)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return None, None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame_num = max(0, min(initial_frame, total_frames - 1))
    
    print("\n--- Manual Frame Adjustment ---")
    print(f"Adjusting: {os.path.basename(video_path)}")
    print("Controls: 'd'/'a' (Next/Prev), Enter (Confirm), 'q' (Quit)")

    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_num)
    ret, frame = cap.read()
    timestamp_ms = 0

    while True:
        if ret:
            display_frame = frame.copy()
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            text = f"Frame: {current_frame_num} @ {timestamp_ms:.2f} ms"
            cv2.putText(display_frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.imshow("Manual Sync Adjustment", display_frame)
        
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27: # 'q' or ESC
            current_frame_num, timestamp_ms, frame = None, None, None
            break
        elif key == 13: # Enter key
            print(f"-> Selected frame: {current_frame_num} at {timestamp_ms:.2f} ms")
            break
        elif key == ord('d') or key == 83: # 'd' or Right arrow
            if current_frame_num < total_frames - 1:
                ret, frame = cap.read()
                if ret: 
                    current_frame_num += 1
        elif key == ord('a') or key == 81: # 'a' or Left arrow
            if current_frame_num > 0:
                new_frame_num = current_frame_num - 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_num)
                ret, frame = cap.read()
                if ret:
                    current_frame_num = new_frame_num
            
    cv2.destroyWindow("Manual Sync Adjustment")
    cap.release()
    return current_frame_num, timestamp_ms, frame


def find_and_fine_tune_sync():
    """
    Main workflow to automatically find, manually adjust, save, and verify sync offsets.
    """
    # --- 1. Select Video Files ---
    root = tk.Tk()
    root.withdraw()

    video_paths = []
    while True:
        print("\nA file dialog will open. Please select a video file for syncing.")
        print("Press 'Cancel' in the dialog when you are finished selecting files.")
        
        video_path = filedialog.askopenfilename(
            title="Select a video file (or Cancel to finish)",
            initialdir="./Videos",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if not video_path: break
        video_paths.append(video_path)
        print(f"  -> Added: {os.path.basename(video_path)}")

    print(f"\nFinished selecting. You chose {len(video_paths)} files.")
    if len(video_paths) < 2:
        print("Error: You must select at least two videos to compare. Exiting.")
        return

    # --- 2. Automatically Find Initial Flash Frame for Each Video ---
    initial_flash_data = []
    print("\nAutomatically analyzing videos for initial sync point...")
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        max_frames_to_check = int(fps * SEARCH_DURATION_SECONDS)
        
        brightest_frame_num, max_brightness, frame_num = -1, -1.0, 0
        while cap.isOpened() and frame_num < max_frames_to_check:
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_brightness = np.mean(gray)
            if current_brightness > max_brightness:
                max_brightness = current_brightness
                brightest_frame_num = frame_num
            frame_num += 1
        
        if brightest_frame_num != -1:
            camera_name = os.path.basename(os.path.dirname(video_path))
            initial_flash_data.append({
                'camera_name': camera_name,
                'path': video_path,
                'flash_frame': brightest_frame_num
            })
            print(f"  - Initial flash for '{camera_name}' found near frame {brightest_frame_num}")
        cap.release()

    if len(initial_flash_data) < 2:
        print("\nError: Could not successfully analyze at least two videos.")
        return

    # --- 3. Manually Adjust Sync Frame for Each Video ---
    adjusted_flash_data = []
    for data in initial_flash_data:
        final_frame_num, final_timestamp, final_frame_image = manually_select_sync_frame(data['path'], data['flash_frame'])
        if final_frame_num is None:
            print("Adjustment cancelled. Exiting.")
            return
        
        adjusted_flash_data.append({
            'camera_name': data['camera_name'],
            'path': data['path'],
            'flash_frame': final_frame_num,
            'sync_timestamp_ms': final_timestamp,
            'verification_image': final_frame_image
        })

    # --- 4. Save New CSV with Timestamps ---
    output_dir = os.path.join("Outputs", "Syncing")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_filename = os.path.join(output_dir, 'sync_offsets_final.csv')
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['camera_name', 'flash_frame', 'sync_timestamp_ms'])
        for data in adjusted_flash_data:
            writer.writerow([data['camera_name'], data['flash_frame'], data['sync_timestamp_ms']])
    print(f"\nFinal sync data saved to '{output_filename}'")

    # --- 5. Final Verification via Static Image ---
    print("\nDisplaying synchronized frames for verification. Press any key to exit.")
    verification_frames = []
    for data in adjusted_flash_data:
        frame = data['verification_image']
        if frame is not None:
            h, w, _ = frame.shape
            scale = DISPLAY_HEIGHT / h
            resized = cv2.resize(frame, (int(w * scale), DISPLAY_HEIGHT))
            text = f"{data['camera_name']}"
            cv2.putText(resized, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            verification_frames.append(resized)
    
    if verification_frames:
        combined_frame = cv2.hconcat(verification_frames)
        cv2.imshow("Final Synchronization Verification", combined_frame)
        cv2.waitKey(0)

    # --- 6. Clean Up ---
    cv2.destroyAllWindows()
    print("Verification finished.")

if __name__ == '__main__':
    find_and_fine_tune_sync()
