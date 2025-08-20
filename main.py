# -*- coding: utf-8 -*-
"""
Master script to orchestrate the entire CanSat 3D tracking workflow.

This script provides a menu-driven interface to run the individual Python
scripts for each step of the process:
1. Calibration (Multiprocess)
2. Tracking (Multiprocess)
3. Synchronization
4. Triangulation

It calls each script as a separate process and provides instructions to the
user for each step.
"""
import subprocess
import os
import time
import tkinter as tk
from tkinter import filedialog
import concurrent.futures
from itertools import repeat

# --- Configuration ---
# Update these names if your script files are named differently.
CALIBRATION_SCRIPT = "calibration.py"
TRACKER_SCRIPT = "realdealtracker.py"
SYNC_SCRIPT = "check_delay.py"
TRIANGULATION_SCRIPT = "triangulation.py"


# --- Helper Functions to Run Scripts ---

def run_script(script_name, file_path=None):
    """
    A helper function to run a python script, optionally passing a file path
    as a command-line argument. This function is defined at the top level
    so it can be 'pickled' by the multiprocessing module.
    """
    if not os.path.exists(script_name):
        print(f"\n--- ERROR ---")
        print(f"Script not found: '{script_name}'")
        return False
    
    command = f"python {script_name}"
    if file_path:
        # Add quotes around the file path to handle spaces
        command += f" \"{file_path}\""

    try:
        print(f"Executing: {command}")
        # Use DEVNULL to hide the output of individual scripts for a cleaner master log
        subprocess.run(command, check=True, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n--- ERROR ---")
        # FIX: Handle cases where file_path is None to prevent TypeError
        error_message = f"An error occurred while running '{script_name}'"
        if file_path:
            error_message += f" with file '{os.path.basename(file_path)}'."
        else:
            error_message += "."
        print(error_message)
        
        # Print the error output from the script if it exists
        if e.stderr:
            print(f"Error details: {e.stderr.decode()}")
        print("---------------")
        return False
    except FileNotFoundError:
        print(f"\n--- ERROR ---")
        print(f"Could not find the 'python' command.")
        return False

def select_files_one_by_one(title):
    """Opens a dialog to select files one by one until the user cancels."""
    video_paths = []
    while True:
        root = tk.Tk()
        root.withdraw()
        print(f"\nA file dialog will open. Please select a video for: {title}")
        print("Press 'Cancel' in the dialog when you are finished selecting files.")
        
        video_path = filedialog.askopenfilename(title=title)
        root.destroy() # Clean up the tkinter window
        
        if not video_path:
            break
        
        video_paths.append(video_path)
        print(f"  -> Added: {os.path.basename(video_path)}")
    
    print(f"\nFinished selecting. You chose {len(video_paths)} file(s).")
    return video_paths


def run_calibration_step(video_paths=None):
    """Guides the user through the multiprocess calibration process."""
    print("\n--- Step 1: Camera Calibration (Parallel) ---")
    
    if not video_paths:
        print("NOTE: For this to work, 'calibration.py' must be modified to accept command-line arguments.")
        video_paths = select_files_one_by_one(title="Select Calibration Video")
    
    if not video_paths:
        print("No videos selected. Skipping calibration.")
        return

    print(f"\nStarting calibration for {len(video_paths)} videos in parallel using all CPU cores...")
    # Use ProcessPoolExecutor for CPU-bound tasks to get true parallelism
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Replaced the lambda function with itertools.repeat to make it pickle-able
        results = list(executor.map(run_script, repeat(CALIBRATION_SCRIPT), video_paths))
    
    # --- Process results and report errors ---
    success_count = 0
    failed_files = []
    for i, success in enumerate(results):
        if success:
            success_count += 1
        else:
            failed_files.append(os.path.basename(video_paths[i]))

    print("\n--- Calibration Step Finished ---")
    print(f"Summary: {success_count} out of {len(video_paths)} tasks completed successfully.")
    if failed_files:
        print("The following files failed to process:")
        for f in failed_files:
            print(f"  - {f}")


def run_tracking_step(video_paths=None):
    """Guides the user through the multiprocess tracking process."""
    print("\n--- Step 2: AprilTag Tracking (Parallel) ---")
    
    if not video_paths:
        print("NOTE: For this to work, 'realdealtracker.py' must be modified to accept command-line arguments.")
        video_paths = select_files_one_by_one(title="Select Video for Tracking")

    if not video_paths:
        print("No videos selected. Skipping tracking.")
        return

    print(f"\nStarting tracking for {len(video_paths)} videos in parallel using all CPU cores...")
    # Use ProcessPoolExecutor for CPU-bound tasks to get true parallelism
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Replaced the lambda function with itertools.repeat
        results = list(executor.map(run_script, repeat(TRACKER_SCRIPT), video_paths))

    # --- Process results and report errors ---
    success_count = 0
    failed_files = []
    for i, success in enumerate(results):
        if success:
            success_count += 1
        else:
            failed_files.append(os.path.basename(video_paths[i]))

    print("\n--- Tracking Step Finished ---")
    print(f"Summary: {success_count} out of {len(video_paths)} tasks completed successfully.")
    if failed_files:
        print("The following files failed to process:")
        for f in failed_files:
            print(f"  - {f}")


def run_sync_step():
    """Runs the synchronization script."""
    print("\n--- Step 3: Video Synchronization ---")
    print("\nThe synchronization script will now open.")
    print("This script will ask you to select your raw video files one by one.")
    time.sleep(2)
    # This script still uses its own dialogs, so we don't pass a file path.
    run_script(SYNC_SCRIPT)
    print("\n--- Synchronization Step Finished ---")


def run_triangulation_step():
    """Runs the final triangulation script."""
    print("\n--- Step 4: 3D Triangulation & Pose Estimation ---")
    print("\nThe final triangulation script will now run.")
    print("This script will ask you to select your 'sync_offsets_final.csv' file.")
    time.sleep(2)
    # This script also uses its own dialogs.
    run_script(TRIANGULATION_SCRIPT)
    print("\n--- Triangulation Step Finished ---")


def run_all_steps_sequentially():
    """Asks for all files upfront, then runs the entire pipeline."""
    print("\n--- Running All Steps Sequentially ---")

    # --- Ask user for video mode ---
    mode_choice = ''
    while mode_choice not in ['S', 'A']:
        mode_choice = input("Use (S)eparate videos for calibration and tracking, or (A)ll-in-one videos? ").upper()

    # --- Collect all file paths upfront based on mode ---
    if mode_choice == 'S':
        print("\n--- Using Separate Videos ---")
        calibration_videos = select_files_one_by_one(title="Select ALL calibration videos")
        if not calibration_videos:
            print("No calibration videos selected. Aborting 'Run All'.")
            return

        raw_videos = select_files_one_by_one(title="Select ALL raw videos for tracking/syncing")
        if not raw_videos:
            print("No raw videos selected. Aborting 'Run All'.")
            return
    else: # mode_choice == 'A'
        print("\n--- Using All-in-One Videos ---")
        all_videos = select_files_one_by_one(title="Select ALL videos for calibration, tracking, and syncing")
        if not all_videos:
            print("No videos selected. Aborting 'Run All'.")
            return
        calibration_videos = all_videos
        raw_videos = all_videos
        
    # --- Execute steps with the collected paths ---
    run_calibration_step(calibration_videos)
    run_tracking_step(raw_videos)
    run_sync_step() # This script still requires manual selection, as it's interactive.
    run_triangulation_step() # This script also requires manual selection.

    print("\n\n--- ALL STEPS COMPLETE ---")
    print("You can find your final data in 'triangulated_pose_data.csv'.")


# --- Main Menu ---

def main():
    """Displays the main menu and runs the selected steps."""
    while True:
        print("\n=============================================")
        print("   CanSat 3D Tracking Workflow Control Panel")
        print("=============================================")
        print("Please choose an option:")
        print("  1. Run Step 1: Camera Calibration (Parallel)")
        print("  2. Run Step 2: AprilTag Tracking (Parallel)")
        print("  3. Run Step 3: Video Synchronization")
        print("  4. Run Step 4: 3D Triangulation")
        print("  -----------------------------------------")
        print("  A. Run All Steps in Order (Interactive Setup)")
        print("  Q. Quit")
        
        choice = input("\nEnter your choice: ").upper()

        if choice == '1':
            run_calibration_step()
        elif choice == '2':
            run_tracking_step()
        elif choice == '3':
            run_sync_step()
        elif choice == '4':
            run_triangulation_step()
        elif choice == 'A':
            run_all_steps_sequentially()
        elif choice == 'Q':
            print("Exiting.")
            break
        else:
            print("Invalid choice, please try again.")
        
        input("\nPress Enter to return to the menu...")


if __name__ == '__main__':
    main()
