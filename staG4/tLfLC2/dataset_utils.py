"""
=========================================================
Dataset Utilities
Emotion Recognition V2
=========================================================
"""

import shutil
from pathlib import Path
from PIL import Image


# -------------------------------------------------------
# Create Folder
# -------------------------------------------------------

def create_folder(folder_path):

    Path(folder_path).mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------
# Resize Image
# -------------------------------------------------------

def resize_image(input_path, output_path, size):

    try:

        image = Image.open(input_path)

        image = image.convert("RGB")

        image = image.resize(size)

        image.save(output_path, quality=95)

        return True

    except Exception as e:

        print(f"Error : {input_path}")

        print(e)

        return False


# -------------------------------------------------------
# Count Images
# -------------------------------------------------------

def count_images(folder):

    total = 0

    for extension in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:

        total += len(list(Path(folder).rglob(extension)))

    return total


# -------------------------------------------------------
# Copy Image
# -------------------------------------------------------

def copy_image(source, destination):

    shutil.copy2(source, destination)


# -------------------------------------------------------
# Check Valid Image
# -------------------------------------------------------

def verify_image(image_path):

    try:

        image = Image.open(image_path)

        image.verify()

        return True

    except Exception:

        return False


# -------------------------------------------------------
# Supported Image
# -------------------------------------------------------

def is_image(file):

    extensions = [

        ".jpg",

        ".jpeg",

        ".png",

        ".bmp"

    ]

    return Path(file).suffix.lower() in extensions