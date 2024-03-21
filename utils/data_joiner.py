#!./venv/bin/python3

import os
import argparse


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file")
    parser.add_argument("--path-to-aug")
    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    # args.path_to_aug = ""
    # args.in_file = ""

    aug_files = [
        f"{args.path_to_aug}/{file}" for file in os.listdir(args.path_to_aug)
        if os.path.isfile(f"{args.path_to_aug}/{file}")
    ]
    out_dir = f"{args.path_to_aug}/aug-org-data"

    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass

    for aug_f in aug_files:
        joined_files = []
        if "augmentation_stats" not in aug_f:
            with open(args.in_file, "r", encoding="utf-8") as in_file:
                joined_files.extend(in_file.readlines())
            with open(aug_f, "r", encoding="utf-8") as aug_file:
                joined_files.extend(aug_file.readlines())
            with open(f"{out_dir}/{aug_f.split('/')[-1]}", "w", encoding="utf-8") as out:
                out.writelines(joined_files)
