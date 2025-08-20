# -*- coding: utf-8 -*-
"""
Generates a chessboard pattern and saves it as a PDF file.

This script is useful for creating calibration patterns for computer vision tasks.
You can customize the board dimensions, square size, and output file name.

Required libraries:
- Pillow (PIL): pip install Pillow
- numpy: pip install numpy
"""
from PIL import Image
import numpy as np

# --- Configuration ---
# You can change these values to customize your chessboard

# Number of inner corners (standard for OpenCV calibration is often 8x6 or 9x7)
# The number of squares will be one greater in each dimension.
corners_x = 8
corners_y = 6

# Size of each square in pixels. A larger value creates a higher-resolution image.
# For printing on A4/Letter paper, 300 pixels is a good starting point (approx 1 inch at 300 DPI).
square_size_px = 300

# Page margins in pixels. This adds a white border around the chessboard.
margin_px = 150

# Output file name
output_filename = "chessboard_pattern.pdf"


# --- Generation Logic ---

def generate_chessboard_pdf():
    """Creates a chessboard image and saves it as a PDF."""
    
    # Calculate the number of squares based on inner corners
    squares_x = corners_x + 1
    squares_y = corners_y + 1

    # Calculate total image dimensions
    board_width = squares_x * square_size_px
    board_height = squares_y * square_size_px
    
    img_width = board_width + 2 * margin_px
    img_height = board_height + 2 * margin_px

    # Create a 3-channel (RGB) numpy array for the image.
    # Initialize with white (255) for the margins.
    img_array = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

    # Generate the chessboard pattern
    # We iterate through each square's top-left pixel position
    for y in range(squares_y):
        for x in range(squares_x):
            # Determine the color of the square (black or white)
            # If (x+y) is even, the square is white. If odd, it's black.
            if (x + y) % 2 == 1:
                color = (0, 0, 0)  # Black
            else:
                color = (255, 255, 255) # White

            # Calculate the pixel coordinates for the current square
            start_x = margin_px + x * square_size_px
            end_x = start_x + square_size_px
            start_y = margin_px + y * square_size_px
            end_y = start_y + square_size_px

            # Fill the square in the numpy array with the calculated color
            img_array[start_y:end_y, start_x:end_x] = color

    # Convert the numpy array to a Pillow Image object
    image = Image.fromarray(img_array, 'RGB')
    
    # Save the image as a PDF
    # The 'resolution' parameter helps ensure good print quality.
    try:
        image.save(output_filename, "PDF", resolution=300.0)
        print(f"Successfully generated '{output_filename}'")
        print(f"Board size: {squares_x}x{squares_y} squares")
        print(f"Inner corners: {corners_x}x{corners_y}")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == '__main__':
    generate_chessboard_pdf()
