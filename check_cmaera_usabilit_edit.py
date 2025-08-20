import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os
import tkinter as tk
from tkinter import filedialog

# --- Configuration ---
# The number of frames to process will be determined by the video length.

def calculate_drift(img1, img2):
    """
    Calculates the spatial drift between two images.
    This is a placeholder function. You should replace this with your
    actual OIS drift calculation logic, which might involve feature matching
    (e.g., SIFT, ORB) or phase correlation.
    For this example, we'll simulate drift with a random value.
    """
    # Placeholder: Simulate drift with a small random offset
    # In a real scenario, this would be a complex calculation.
    simulated_drift = np.random.uniform(0, 1.5)
    return simulated_drift

def analyze_ois_drift():
    """
    Analyzes a video file to calculate OIS drift, plots the results,
    and saves the data.
    """
    # --- Get Video File Path using a File Dialog ---
    # Set up the root window for tkinter
    root = tk.Tk()
    root.withdraw() # Hide the main window

    # Open the file selection dialog
    video_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=(("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*"))
    )

    # If the user cancels the dialog, video_path will be empty
    if not video_path:
        print("No file selected. Exiting.")
        return

    # Check if the file exists before attempting to open it
    if not os.path.exists(video_path):
        print(f"Error: The file '{video_path}' was not found.")
        return

    # --- Initialize Video Capture ---
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return

    # Get total number of frames in the video
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video file loaded successfully. Total frames: {num_frames}")
    print("Starting OIS drift analysis...")


    # Lists to store data for plotting and calculation
    drift_values = []
    frame_numbers = []
    
    # Capture the initial frame to compare against
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Failed to capture the first frame from the video.")
        cap.release()
        return
        
    # Convert the first frame to grayscale for calculations
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Main loop to capture frames and calculate drift
    frame_count = 1
    while True:
        ret, current_frame = cap.read()
        # If 'ret' is False, it means we've reached the end of the video
        if not ret:
            break

        # Convert the current frame to grayscale
        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Calculate the drift between the initial and current frame
        drift = calculate_drift(prev_frame_gray, current_frame_gray)
        
        # Store the calculated drift and frame number
        drift_values.append(drift)
        frame_numbers.append(frame_count)
        
        frame_count += 1

    # --- Analysis and Output ---

    # Calculate the average drift
    if drift_values:
        average_drift = np.mean(drift_values)
        print("\n-----------------------------------------")
        print(f"Analysis Complete.")
        print(f"Average Drift: {average_drift:.4f} pixels")
        print("-----------------------------------------")
    else:
        print("\nAnalysis failed: No drift values were recorded.")
        cap.release()
        return

    # --- Create Directory for Results ---
    # Define the main parent directory for all results
    results_parent_dir = 'results'
    
    # Create a directory named after the video file, inside the parent results directory
    video_filename = os.path.basename(video_path)
    filename_without_ext = os.path.splitext(video_filename)[0]
    video_results_folder = f'{filename_without_ext}_results'
    
    # The full path for this specific video's results folder
    results_dir = os.path.join(results_parent_dir, video_results_folder)
    
    try:
        # Use os.makedirs to create the parent and child directories if they don't exist
        os.makedirs(results_dir, exist_ok=True)
        print(f"Saving results to '{results_dir}/' directory.")
    except OSError as e:
        print(f"Error creating directory {results_dir}: {e}")
        # Fallback to current directory if creation fails
        results_dir = '.'

    # --- Plotting the Drift ---
    
    print("Generating drift plot...")
    plt.figure(figsize=(12, 6))
    plt.plot(frame_numbers, drift_values, marker='o', linestyle='-', markersize=2, label='OIS Drift')
    # Add a line for the average drift
    plt.axhline(y=average_drift, color='r', linestyle='--', label=f'Average Drift ({average_drift:.4f})')
    
    # Add titles and labels for clarity
    plt.title('OIS Drift Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Drift (pixels)')
    plt.grid(True)
    plt.legend()
    
    # Save the plot to a file in the results directory
    plot_filename = os.path.join(results_dir, 'ois_drift_plot.png')
    try:
        plt.savefig(plot_filename)
        print(f"Successfully saved plot to '{plot_filename}'")
    except Exception as e:
        print(f"Error: Could not save plot. {e}")
    
    # --- Saving the Data ---

    print("Saving drift data to file...")
    # Save the data file in the results directory with a .csv extension
    data_filename = os.path.join(results_dir, 'ois_drift_data.csv')
    try:
        with open(data_filename, 'w') as f:
            f.write("Frame,Drift\n") # Write a header
            for frame, drift in zip(frame_numbers, drift_values):
                f.write(f"{frame},{drift:.4f}\n")
        print(f"Successfully saved data to '{data_filename}'")
    except Exception as e:
        print(f"Error: Could not save data file. {e}")

    # Release the video capture object
    cap.release()
    print("\nProcess finished.")


if __name__ == '__main__':
    # Ensure you have the required libraries installed:
    # pip install numpy opencv-python matplotlib
    analyze_ois_drift()
