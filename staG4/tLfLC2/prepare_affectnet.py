"""
=========================================================
Prepare AffectNet Dataset (YOLO Format)
Emotion Recognition V2
=========================================================
"""

import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import config


# ==========================================================
# UPDATE THIS AFTER CHECKING data.yaml
# ==========================================================

CLASS_MAP = {

    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise"

}


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

    for split in ["train", "valid", "test"]:

        for emotion in config.EMOTIONS:

            folder = (

                config.OUTPUT_DATASET /

                "AFFECT" /

                split /

                emotion

            )

            folder.mkdir(

                parents=True,

                exist_ok=True

            )


# ==========================================================
# READ LABEL
# ==========================================================

def read_label(label_path):

    try:

        with open(label_path, "r") as file:

            line = file.readline().strip()

        values = line.split()

        class_id = int(values[0])

        return class_id

    except Exception:

        return None


# ==========================================================
# COPY DATASET
# ==========================================================

def copy_dataset():

    copied = 0

    skipped = 0

    for split in [

        "train",

        "valid",

        "test"

    ]:

        print(f"\nProcessing {split.upper()}")

        image_folder = (

            config.AFFECTNET_PATH /

            split /

            "images"

        )

        label_folder = (

            config.AFFECTNET_PATH /

            split /

            "labels"

        )

        images = []

        for ext in [

            "*.jpg",

            "*.jpeg",

            "*.png"

        ]:

            images.extend(

                image_folder.glob(ext)

            )

        for index, image_path in enumerate(tqdm(images)):

            label_path = (

                label_folder /

                f"{image_path.stem}.txt"

            )

            if not label_path.exists():

                skipped += 1

                continue

            class_id = read_label(label_path)

            if class_id is None:

                skipped += 1

                continue

            if class_id not in CLASS_MAP:

                skipped += 1

                continue

            emotion = CLASS_MAP[class_id]

            if not verify_image(image_path):

                skipped += 1

                continue

            destination = (

                config.OUTPUT_DATASET /

                "AFFECT" /

                split /

                emotion

            )

            filename = (

                f"AFFECT_{emotion}_{index:06d}.jpg"

            )

            shutil.copy2(

                image_path,

                destination / filename

            )

            copied += 1

    print("\n==============================")

    print("AffectNet Completed")

    print(f"Copied : {copied}")

    print(f"Skipped : {skipped}")

    print("==============================")


# ==========================================================
# SUMMARY
# ==========================================================

def summary():

    print("\nDataset Summary\n")

    root = config.OUTPUT_DATASET / "AFFECT"

    grand_total = 0

    for split in [

        "train",

        "valid",

        "test"

    ]:

        print(f"\n{split.upper()}")

        split_total = 0

        for emotion in config.EMOTIONS:

            folder = root / split / emotion

            count = len(list(folder.glob("*")))

            split_total += count

            print(f"{emotion:10s}: {count}")

        print(f"Total : {split_total}")

        grand_total += split_total

    print("\n==============================")

    print(f"Grand Total : {grand_total}")

    print("==============================")


# ==========================================================
# MAIN
# ==========================================================

def main():

    create_folders()

    copy_dataset()

    summary()


if __name__ == "__main__":

    main()