"""
=========================================================
Merge FER + RAF + AffectNet
Emotion Recognition V2
=========================================================
"""

import shutil
from pathlib import Path

import config

# ==========================================================
# COPY IMAGES
# ==========================================================

def copy_dataset(dataset_name):

    source = config.OUTPUT_DATASET / dataset_name

    if not source.exists():

        print(f"{dataset_name} Not Found")

        return

    print(f"\nCopying {dataset_name}")

    for split in source.iterdir():

        if not split.is_dir():

            continue

        split_name = split.name

        for emotion in split.iterdir():

            if not emotion.is_dir():

                continue

            destination = (

                config.FINAL_DATASET /

                split_name /

                emotion.name

            )

            destination.mkdir(

                parents=True,

                exist_ok=True

            )

            images = list(

                emotion.glob("*")

            )

            print(

                f"{split_name}/{emotion.name} : {len(images)}"

            )

            for image in images:

                shutil.copy2(

                    image,

                    destination / image.name

                )


# ==========================================================
# SUMMARY
# ==========================================================

def summary():

    print("\n====================================")

    print("FINAL DATASET")

    print("====================================")

    total = 0

    for split in [

        "train",

        "valid",

        "test"

    ]:

        split_path = (

            config.FINAL_DATASET /

            split

        )

        if not split_path.exists():

            continue

        print(f"\n{split.upper()}")

        split_total = 0

        for emotion in sorted(split_path.iterdir()):

            count = len(

                list(

                    emotion.glob("*")

                )

            )

            split_total += count

            print(

                f"{emotion.name:10s}: {count}"

            )

        print(

            f"Total : {split_total}"

        )

        total += split_total

    print("\n====================================")

    print(f"Grand Total : {total}")

    print("====================================")


# ==========================================================
# MAIN
# ==========================================================

def main():

    if config.USE_FER:

        copy_dataset("FER")

    if config.USE_RAF:

        copy_dataset("RAF")

    if config.USE_AFFECT:

        copy_dataset("AFFECT")

    summary()


if __name__ == "__main__":

    main()