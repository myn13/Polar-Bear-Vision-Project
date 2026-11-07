import numpy as np
import os
import re
from PIL import Image

def parse_file(filename):
    " eg: 000C01T000F012.jpg"
    " pattern: {id}C{camera id}T{tracklet id}F{frame id}"
    pattern = r"^(\d{3})C(\d{2})T(\d{3})F(\d{3})\.jpg$"
    match_pattern = re.search(pattern, filename)
    if match_pattern:
        return {
            "ID": int(match_pattern.group(1)),
            "camera_id": int(match_pattern.group(2)),
            "tracklet_id": int(match_pattern.group(3)),
            "frame_id": int(match_pattern.group(4)),
        }
    else:
        return None

def preprocess_image_to_numpy(image_path, target_size = (224, 224)):
    filename = os.path.basename(image_path)
    file_info = parse_file(filename)
    print(file_info)
    try:
        with Image.open(image_path) as image:
            image = image.convert('RGB')
            image = image.resize(target_size)
            img_array = np.array(image, dtype=np.float32)
            img_array /= 255.0
            return img_array
    except Exception as e:
        print("Error in preprocessing {image_path}: {e}")
        return None

def explore_directory(root):
    if not os.path.exists(root):
        print(f"Directory {root} not exist")
        return
    for root, dirs, files in os.walk(root):
            for f in files:
                if f.lower().endswith('.csv'):
                    continue

                if f.lower().endswith('.jpg'):
                    filepath = os.path.join(root, f)
                    print(f"Found {f}")
                    preprocess_image_to_numpy(filepath)
