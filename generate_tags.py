# generate_tags.py
# Description: A standalone script to generate AprilTag marker images.
# This final version corrects the data orientation to match the official specification,
# ensuring the generated tags are valid and detectable.
#
# To run:
# 1. Install dependencies: pip install Pillow requests
# 2. Run the script, e.g.:
#    python generate_tags.py

import argparse
import os
import re
import importlib.util
from PIL import Image

try:
    import requests
except ImportError:
    print("Error: 'requests' library not found. Please run 'pip install requests'")
    exit()

# --- Tag Family Data Management ---

def fetch_and_parse_tag_family(family_name):
    """
    Fetches the C source file for a tag family from the official GitHub repo,
    parses the codes, and saves them to a local Python file.
    """
    url = f"https://raw.githubusercontent.com/AprilRobotics/apriltag/master/apriltag/common/tag/tagFamilies/{family_name}.c"
    print(f"Downloading official tag data for {family_name} from {url}...")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        c_code = response.text

        match = re.search(r"static const uint64_t codes\[\] = \{(.*?)\};", c_code, re.DOTALL)
        if not match:
            raise ValueError("Could not find the codes array in the downloaded file.")

        codes_str = match.group(1)
        codes = [int(code.strip(), 16) for code in codes_str.split(',') if code.strip()]
        
        with open(f"{family_name}.py", "w", encoding="utf-8") as f:
            f.write(f"# Official AprilTag codes for {family_name}\n")
            f.write(f"codes = {codes}\n")
        
        print(f"Successfully downloaded and saved data to {family_name}.py")
        return {"codes": codes}

    except requests.exceptions.RequestException as e:
        print(f"Error: Could not download tag data. Please check your internet connection. Details: {e}")
        return None

def load_tag_family(family_name):
    """
    Loads tag family data, either from a local cache file or by downloading it.
    """
    family_file = f"{family_name}.py"
    if os.path.exists(family_file):
        print(f"Loading tag data from local file: {family_file}")
        spec = importlib.util.spec_from_file_location(family_name, family_file)
        family_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(family_module)
        return {"codes": family_module.codes}
    else:
        return fetch_and_parse_tag_family(family_name)

# --- Tag Generation (Rewritten Logic) ---

def generate_tag(family_data, tag_id, pixel_size=20, quiet_zone=2):
    """
    Generates a PIL Image for a given AprilTag with the correct 10x10 structure.
    """
    codes = family_data["codes"]
    if not (0 <= tag_id < len(codes)):
        raise ValueError(f"Tag ID {tag_id} is out of range (0-{len(codes)-1})")

    tag_code = codes[tag_id]
    
    tag_width = 10
    
    # --- FINAL CORRECTED LOGIC ---
    # This builds the tag from a simple grid, ensuring correct bit order and orientation.
    # 0 = black, 1 = white
    grid = [[0] * tag_width for _ in range(tag_width)] # Start with a black 10x10 grid

    # 1. Create the inner white border (quiet zone)
    for y in range(1, tag_width - 1):
        for x in range(1, tag_width - 1):
            grid[y][x] = 1

    # 2. Place the 36 data bits in their correct positions
    # The bits in the code are ordered from most significant to least significant.
    # The most significant bit corresponds to the top-left of the data grid.
    bit_index = 35 
    for y_data in range(6):
        for x_data in range(6):
            # Check the bit
            bit = (tag_code >> bit_index) & 1
            
            # The +2 offset places the 6x6 data grid inside the borders
            grid_x = x_data + 2
            grid_y = y_data + 2

            # A '1' bit in the code means a WHITE square, '0' is BLACK
            grid[grid_y][grid_x] = 1 if bit == 1 else 0
            
            bit_index -= 1 # Move to the next bit

    # --- Create the final image from the grid ---
    canvas_width = tag_width + 2 * quiet_zone
    image_size = canvas_width * pixel_size
    img = Image.new('L', (image_size, image_size), color=255) # White background
    pixels = img.load()

    for y_grid in range(tag_width):
        for x_grid in range(tag_width):
            if grid[y_grid][x_grid] == 0: # If grid cell is black
                x_start = (x_grid + quiet_zone) * pixel_size
                y_start = (y_grid + quiet_zone) * pixel_size
                for py in range(pixel_size):
                    for px in range(pixel_size):
                        pixels[x_start + px, y_start + py] = 0
    return img

def main():
    parser = argparse.ArgumentParser(description="Generate AprilTag markers.")
    parser.add_argument("--family", type=str, default="tag36h11", help="Tag family name.")
    parser.add_argument("--start", type=int, default=0, help="Starting tag ID.")
    parser.add_argument("--end", type=int, default=10, help="Ending tag ID (inclusive).")
    parser.add_argument("--output", type=str, default="./tags", help="Output directory.")
    parser.add_argument("--quiet_zone", type=int, default=2, help="Width of the extra white border in cells.")
    
    args = parser.parse_args()

    family_data = load_tag_family(args.family)
    if not family_data:
        exit()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f"Created directory: {args.output}")

    print(f"Generating tags for family '{args.family}' from ID {args.start} to {args.end}...")

    for tag_id in range(args.start, args.end + 1):
        try:
            tag_image = generate_tag(family_data, tag_id, quiet_zone=args.quiet_zone)
            filename = f"{args.family}_{tag_id:05d}.png"
            filepath = os.path.join(args.output, filename)
            tag_image.save(filepath)
            print(f"  - Saved {filepath}")
        except ValueError as e:
            print(f"Error generating tag {tag_id}: {e}")
            break
            
    print("Done.")

if __name__ == "__main__":
    main()
