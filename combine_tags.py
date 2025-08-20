# combine_tags.py
# Description: A script to combine multiple AprilTag images from a directory
# into a single PDF sheet for easy printing.
#
# To run:
# 1. Make sure this script is in the same folder as your tag images.
# 2. Install Pillow: pip install Pillow
# 3. Run the script: python combine_tags.py

import os
from PIL import Image
import glob

# --- Configuration ---
# Directory containing the tag images. '.' means the current directory.
IMAGE_DIRECTORY = './tags'
# Pattern to find the tag files. '*' is a wildcard.
FILE_PATTERN = 'tag36h11_*.png'
# Name of the final output file.
OUTPUT_FILENAME = 'printable_tag_sheet.pdf'

# Layout settings
# You can change these to adjust the final look.
IMAGES_PER_ROW = 5
PADDING = 20  # Pixels of white space around each tag

def combine_images():
    """Finds all tag images and combines them into a single sheet."""
    # Find all files in the directory that match the pattern
    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIRECTORY, FILE_PATTERN)))

    if not image_paths:
        print(f"Error: No images found matching '{FILE_PATTERN}' in the current directory.")
        return

    print(f"Found {len(image_paths)} tags to combine.")

    # Open the first image to get dimensions
    with Image.open(image_paths[0]) as img:
        img_width, img_height = img.size

    # Calculate the dimensions of the final combined image
    num_images = len(image_paths)
    num_rows = (num_images + IMAGES_PER_ROW - 1) // IMAGES_PER_ROW # Ceiling division
    
    sheet_width = (img_width * IMAGES_PER_ROW) + (PADDING * (IMAGES_PER_ROW + 1))
    sheet_height = (img_height * num_rows) + (PADDING * (num_rows + 1))

    # Create a new blank white image to be the canvas
    tag_sheet = Image.new('L', (sheet_width, sheet_height), color=255)
    print(f"Creating a printable sheet of size {sheet_width}x{sheet_height} pixels.")

    # Paste each tag onto the sheet
    for i, image_path in enumerate(image_paths):
        row = i // IMAGES_PER_ROW
        col = i % IMAGES_PER_ROW

        # Calculate the top-left corner position for the current tag
        x_pos = (col * img_width) + (PADDING * (col + 1))
        y_pos = (row * img_height) + (PADDING * (row + 1))

        with Image.open(image_path) as img:
            tag_sheet.paste(img, (x_pos, y_pos))
    
    # --- Save as PDF ---
    # Convert the grayscale image ('L') to RGB, which is better for PDF compatibility.
    pdf_canvas = tag_sheet.convert('RGB')
    
    # Save the final image as a PDF
    pdf_canvas.save(OUTPUT_FILENAME, "PDF", resolution=100.0)
    print(f"\nSuccessfully created '{OUTPUT_FILENAME}'. You can now print this file.")

if __name__ == "__main__":
    combine_images()