import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset")
    args = parser.parse_args()

    img_dir = os.path.join(args.data_dir, "images")
    mask_dir = os.path.join(args.data_dir, "masks")

    img_files = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir)}
    mask_files = {os.path.splitext(f)[0]: f for f in os.listdir(mask_dir)}

    img_stems = set(img_files.keys())
    mask_stems = set(mask_files.keys())

    matched = img_stems & mask_stems
    only_in_images = img_stems - mask_stems
    only_in_masks = mask_stems - img_stems

    print(f"Total di images/            : {len(img_stems)}")
    print(f"Total di masks/             : {len(mask_stems)}")
    print(f"Pasangan cocok (akan dipakai train.py) : {len(matched)}")
    print(f"Hanya ada di images/ (TIDAK terpakai)  : {len(only_in_images)}")
    print(f"Hanya ada di masks/  (TIDAK terpakai)   : {len(only_in_masks)}")

    if only_in_images:
        print("\nContoh 10 file di images/ yang TIDAK punya pasangan mask:")
        for stem in list(only_in_images)[:10]:
            print(f"  - {img_files[stem]}")

    if only_in_masks:
        print("\nContoh 10 file di masks/ yang TIDAK punya pasangan image:")
        for stem in list(only_in_masks)[:10]:
            print(f"  - {mask_files[stem]}")

if __name__ == "__main__":
    main()