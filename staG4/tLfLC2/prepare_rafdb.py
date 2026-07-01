"""
=========================================================
Prepare RAF-DB Dataset
Emotion Recognition V2
=========================================================
"""

import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import config


# ==========================================================
# VERIFY IMAGE
# ==========================================================

def verify_image(image_path):

    try:
        img = Image.open(image_path)
        img.verify()
        return True

    except Exception:
        return False


# ==========================================================
# CREATE OUTPUT FOLDERS
# ==========================================================

def create_folders():

    for split in ["train", "test"]:

        for emotion in config.EMOTIONS:

            folder = (
                config.OUTPUT_DATASET /
                "RAF" /
                split /
                emotion
            )

            folder.mkdir(
                parents=True,
                exist_ok=True
            )


# ==========================================================
# COPY DATASET
# ==========================================================

def copy_dataset():

    print("\nPreparing RAF-DB Dataset...\n")

    copied = 0
    skipped = 0

    dataset_root = config.RAF_PATH / "DATASET"

    for split in ["train", "test"]:

        split_path = dataset_root / split

        for label_folder in sorted(split_path.iterdir()):

            if not label_folder.is_dir():
                continue

            label = label_folder.name

            if label not in config.RAF_LABELS:

                print(f"Skipping Unknown Label {label}")

                continue

            emotion = config.RAF_LABELS[label]

            destination_folder = (
                config.OUTPUT_DATASET /
                "RAF" /
                split /
                emotion
            )

            images = []

            for ext in [

                "*.jpg",
                "*.jpeg",
                "*.png",
                "*.bmp"

            ]:

                images.extend(
                    label_folder.glob(ext)
                )

            print(f"\n{split.upper()} -> {emotion}")
            print(f"Images : {len(images)}")

            for index, image_path in enumerate(tqdm(images)):

                if not verify_image(image_path):

                    skipped += 1
                    continue

                new_name = (
                    f"RAF_{emotion}_{index:06d}.jpg"
                )

                destination = (
                    destination_folder /
                    new_name
                )

                shutil.copy2(
                    image_path,
                    destination
                )

                copied += 1

    print("\n===================================")
    print("RAF-DB Preparation Completed")
    print("===================================")

    print(f"Copied Images : {copied}")
    print(f"Skipped Images: {skipped}")


# ==========================================================
# SUMMARY
# ==========================================================

def summary():

    print("\nDataset Summary\n")

    root = config.OUTPUT_DATASET / "RAF"

    grand_total = 0

    for split in ["train", "test"]:

        print(f"\n{split.upper()}")

        split_total = 0

        for emotion in config.EMOTIONS:

            folder = root / split / emotion

            count = len(list(folder.glob("*")))

            split_total += count

            print(f"{emotion:10s}: {count}")

        print(f"Total : {split_total}")

        grand_total += split_total

    print("\n===================================")
    print(f"Grand Total : {grand_total}")
    print("===================================")


# ==========================================================
# MAIN
# ==========================================================

def main():

    create_folders()

    copy_dataset()

    summary()


if __name__ == "__main__":

    main()