"""
=========================================================
Emotion Recognition V2
Dataset Analyzer
=========================================================
"""

from pathlib import Path
from collections import Counter
from PIL import Image

import config


# =====================================================
# PRINT TITLE
# =====================================================

def title(name):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)


# =====================================================
# ANALYZE FER2013
# =====================================================

def analyze_fer():

    title("FER2013")

    train = config.FER_PATH / "train"
    test = config.FER_PATH / "test"

    total = 0

    for folder in sorted(train.iterdir()):

        if folder.is_dir():

            count = len(list(folder.glob("*")))

            total += count

            print(f"{folder.name:12s}: {count}")

    print("-" * 40)
    print(f"Training Images : {total}")

    total = 0

    for folder in sorted(test.iterdir()):

        if folder.is_dir():

            count = len(list(folder.glob("*")))

            total += count

            print(f"{folder.name:12s}: {count}")

    print("-" * 40)
    print(f"Testing Images : {total}")


# =====================================================
# ANALYZE RAFDB
# =====================================================

def analyze_raf():

    title("RAF-DB")

    dataset = config.RAF_PATH / "DATASET"

    for split in ["train", "test"]:

        print(f"\n{split.upper()}")

        path = dataset / split

        total = 0

        for folder in sorted(path.iterdir()):

            if folder.is_dir():

                count = len(list(folder.glob("*")))

                total += count

                print(

                    f"{folder.name:5s}: {count}"

                )

        print(f"Total : {total}")


# =====================================================
# ANALYZE AFFECTNET
# =====================================================

def analyze_affectnet():

    title("AffectNet")

    for split in [

        "train",

        "valid",

        "test"

    ]:

        print(f"\n{split.upper()}")

        images = config.AFFECTNET_PATH / split / "images"

        labels = config.AFFECTNET_PATH / split / "labels"

        print(

            "Images :", len(list(images.glob("*")))

        )

        print(

            "Labels :", len(list(labels.glob("*")))

        )

        sample = list(labels.glob("*"))[:5]

        print("\nSample Label Files")

        for s in sample:

            print(s.name)

            try:

                print(

                    s.read_text().strip()

                )

            except:

                print("Cannot Read")

            print("-" * 20)


# =====================================================
# IMAGE SIZE
# =====================================================

def image_size():

    title("IMAGE SIZE")

    sample = list(

        config.FER_PATH.rglob("*.jpg")

    )[:5]

    for image in sample:

        img = Image.open(image)

        print(

            image.name,

            img.size

        )


# =====================================================
# MAIN
# =====================================================

def main():

    analyze_fer()

    analyze_raf()

    analyze_affectnet()

    image_size()


if __name__ == "__main__":

    main()