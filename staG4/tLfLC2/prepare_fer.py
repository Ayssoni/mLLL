"""
=========================================================
Prepare FER2013 Dataset
Emotion Recognition V2
=========================================================
"""

import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import config


# =====================================================
# VERIFY IMAGE
# =====================================================

def verify_image(image_path):

    try:
        img = Image.open(image_path)
        img.verify()
        return True

    except Exception:
        return False


# =====================================================
# CREATE OUTPUT FOLDERS
# =====================================================

def create_folders():

    for split in ["train", "test"]:

        for emotion in config.EMOTIONS:

            folder = (
                config.OUTPUT_DATASET
                / "FER"
                / split
                / emotion
            )

            folder.mkdir(
                parents=True,
                exist_ok=True
            )


# =====================================================
# COPY DATASET
# =====================================================

def copy_dataset():

    print("\nPreparing FER2013 Dataset...\n")

    total_images = 0
    skipped = 0

    for split in ["train", "test"]:

        source_split = config.FER_PATH / split

        for emotion in config.EMOTIONS:

            source_folder = source_split / emotion

            destination_folder = (
                config.OUTPUT_DATASET
                / "FER"
                / split
                / emotion
            )

            images = []

            for extension in [

                "*.jpg",
                "*.jpeg",
                "*.png",
                "*.bmp"

            ]:

                images.extend(
                    source_folder.glob(extension)
                )

            print(f"\n{split.upper()} -> {emotion}")
            print(f"Images : {len(images)}")

            for index, image_path in enumerate(
                tqdm(images)
            ):

                if not verify_image(image_path):

                    skipped += 1
                    continue

                new_name = (
                    f"FER_{emotion}_{index:06d}.jpg"
                )

                destination = (
                    destination_folder / new_name
                )

                shutil.copy2(
                    image_path,
                    destination
                )

                total_images += 1

    print("\n====================================")
    print("FER2013 Preparation Completed")
    print("====================================")

    print(f"Copied Images : {total_images}")
    print(f"Skipped Images: {skipped}")


# =====================================================
# DATASET SUMMARY
# =====================================================

def summary():

    print("\nDataset Summary\n")

    root = config.OUTPUT_DATASET / "FER"

    total = 0

    for split in ["train", "test"]:

        print(f"\n{split.upper()}")

        split_total = 0

        for emotion in config.EMOTIONS:

            folder = root / split / emotion

            count = len(list(folder.glob("*")))

            split_total += count

            print(
                f"{emotion:10s}: {count}"
            )

        print(
            f"Total : {split_total}"
        )

        total += split_total

    print("\n====================================")

    print(f"Grand Total : {total}")

    print("====================================")


# =====================================================
# MAIN
# =====================================================

def main():

    create_folders()

    copy_dataset()

    summary()


if __name__ == "__main__":

    main()