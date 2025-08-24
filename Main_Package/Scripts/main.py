# -*- coding: utf-8 -*-
"""
Master script to orchestrate the entire CanSat 3D tracking workflow.

This script provides a menu-driven interface to run the individual Python
scripts for each step of the process:
1. Camera Calibration (Multiprocess)
2. Object Calibration
3. Tracking (Multiprocess)
4. Syncing
5. Camera Triangulation
6. Tag Filling
7. Object Triangulation
8. Visualization

It calls each script as a separate process and manages the file paths
to create a seamless, automated pipeline. It also includes a save/resume
feature for the 'Run All' mode.
"""
import subprocess
import os
import time
import tkinter as tk
from tkinter import filedialog
import concurrent.futures
from itertools import repeat
import json

# --- Configuration ---
# This dictionary maps each step to its corresponding script.
SCRIPTS = {
    "camera_calibration": "Scripts/camera_calibration.py",
    "object_calibration": "Scripts/object_calibration.py",
    "tracking": "Scripts/tracking.py",
    "syncing": "Scripts/syncing.py",
    "camera_triangulation": "Scripts/camera_triangulation.py",
    "tag_filler": "Scripts/tag_filler.py",
    "object_triangulation": "Scripts/object_triangulation.py",
    "visualizer": "Scripts/final_visualizer.html"
}
SAVE_FILE = "workflow_save.json"

# --- Helper Functions ---

def run_script(script_name, file_path=None):
    """Runs a Python script as a separate process."""
    command = f"python {script_name}"
    if file_path:
        command += f' "{file_path}"'
    
    try:
        print(f"Executing: {command}")
        subprocess.run(command, check=True, shell=True, capture_output=True, text=True)
        return True, ""
    except subprocess.CalledProcessError as e:
        error_message = f"Error in '{script_name}':\n{e.stderr}"
        print(error_message)
        return False, error_message

def run_parallel(script_name, file_paths):
    """Runs a script on multiple files in parallel using multiprocessing."""
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(run_script, repeat(script_name), file_paths))
    
    failed_files = [os.path.basename(file_paths[i]) for i, (success, _) in enumerate(results) if not success]
    return not any(failed_files), failed_files

def select_files(title, initial_dir, file_types):
    """Opens a file dialog for the user to select one or more files."""
    root = tk.Tk()
    root.withdraw()
    files = filedialog.askopenfilenames(title=title, initialdir=initial_dir, filetypes=file_types)
    root.destroy()
    return list(files)

def save_progress(step_index, file_paths):
    """Saves the current state of the workflow to a file."""
    with open(SAVE_FILE, 'w') as f:
        json.dump({"last_completed_step": step_index, "file_paths": file_paths}, f)
    print(f"Progress saved. You can resume from step {step_index + 2} later.")

def load_progress():
    """Loads the workflow state from a file if it exists."""
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, 'r') as f:
            return json.load(f)
    return None

def clear_progress():
    """Removes the save file."""
    if os.path.exists(SAVE_FILE):
        os.remove(SAVE_FILE)

# --- Main Workflow Steps ---

def run_all_steps():
    """Runs the entire pipeline from start to finish."""
    progress = load_progress()
    start_step = 0
    file_paths = {}

    if progress:
        resume = input(f"Found saved progress. Resume from step {progress['last_completed_step'] + 2}? (y/n): ").lower()
        if resume == 'y':
            start_step = progress['last_completed_step'] + 1
            file_paths = progress['file_paths']
        else:
            clear_progress()

    if start_step == 0:
        print("\n--- Starting New Workflow ---")
        # Step 1 & 2: Get video files
        file_paths['calibration'] = select_files("Select ALL calibration videos", "./Videos", [("Video Files", "*.mp4")])
        if not file_paths['calibration']: return
        
        file_paths['object_cal'] = select_files("Select the object calibration video", "./Videos", [("Video Files", "*.mp4")])
        if not file_paths['object_cal']: return

        file_paths['tracking'] = select_files("Select ALL main tracking videos", "./Videos", [("Video Files", "*.mp4")])
        if not file_paths['tracking']: return

    
    steps = [
        ("Camera Calibration", lambda: run_parallel(SCRIPTS["camera_calibration"], file_paths['calibration'])),
        ("Object Calibration", lambda: run_script(SCRIPTS["object_calibration"], file_paths['object_cal'][0])),
        ("Tracking", lambda: run_parallel(SCRIPTS["tracking"], file_paths['tracking'])),
        ("Syncing", lambda: run_script(SCRIPTS["syncing"])),
        ("Camera Triangulation", lambda: run_script(SCRIPTS["camera_triangulation"])),
        ("Tag Filling", lambda: run_script(SCRIPTS["tag_filler"])),
        ("Object Triangulation", lambda: run_script(SCRIPTS["object_triangulation"]))
    ]

    for i, (name, func) in enumerate(steps):
        if i < start_step:
            continue
        
        print(f"\n--- Running Step {i+1}: {name} ---")
        success, result = func()
        
        if not success:
            print(f"Step '{name}' failed. Aborting workflow.")
            quit_choice = input("Press 'q' to quit or any other key to return to the main menu: ").lower()
            if quit_choice == 'q':
                exit()
            return
        
        save_progress(i, file_paths)

    # Final Visualization Step
    viz_choice = input("\nAll calculations are complete. Do you want to run the 3D visualizer? (y/n): ").lower()
    if viz_choice == 'y':
        print("Opening visualizer... Close the browser tab to return to the menu.")
        os.startfile(os.path.abspath(SCRIPTS["visualizer"]))
    
    clear_progress() # Clear save file on successful completion
    print("\n--- Workflow Complete! ---")


# --- Main Menu ---

def main():
    while True:
        print("\n" + "="*50)
        print("   CanSat 3D Tracking Workflow Control Panel")
        print("="*50)
        print("  1. Run All Steps (Recommended)")
        print("  2. Run a single step")
        print("  Q. Quit")
        
        choice = input("\nEnter your choice: ").upper()

        if choice == '1':
            run_all_steps()
        elif choice == '2':
            print("\nSelect a single step to run:")
            print("  1. Camera Calibration")
            print("  2. Object Calibration")
            # ... add other single steps if needed
            step_choice = input("Enter step number: ")
            if step_choice == '1':
                videos = select_files("Select calibration videos", "./Videos", [("Video Files", "*.mp4")])
                run_parallel(SCRIPTS["camera_calibration"], videos)
            elif step_choice == '2':
                video = select_files("Select object calibration video", "./Videos", [("Video Files", "*.mp4")])
                if video: run_script(SCRIPTS["object_calibration"], video[0])

        elif choice == 'Q':
            break
        else:
            print("Invalid choice.")
        
        input("\nPress Enter to return to the menu...")

if __name__ == '__main__':
    # This check is crucial for multiprocessing to work correctly on Windows
    main()
