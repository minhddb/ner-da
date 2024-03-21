#!./venv/bin/python3
import os

from augmentation import Augmentation
from dataset import to_tsv, to_json

import itertools
import argparse


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str)
    parser.add_argument("--output-path", type=str, help="Path to folder to store output files")
    parser.add_argument("--word-column", type=int, default=0, help="Index of words column")
    parser.add_argument("--tag-columns",
                        type=int,
                        nargs="+",
                        default=None,
                        help="Specify entities (tags) column(s) for data augmentation")
    parser.add_argument("--main-entity-column",
                        type=int,
                        default=1,
                        help="Specify the main task entity column.")
    parser.add_argument("--segment-based-augmentation", action="store_true")
    parser.add_argument("--character-based-augmentation", action="store_true")
    parser.add_argument("--p-augmentation",
                        type=float,
                        default=0.5,
                        help="Probability for for binomial distribution")
    parser.add_argument("--seed",
                        type=int,
                        default=42)
    parser.add_argument("--to-tsv", action="store_true")
    parser.add_argument("--to-json", action="store_true")
    parser.add_argument("--json-columns",
                        type=str,
                        nargs="+",
                        default=None,
                        help="Optional columns for json file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()

    try:
        os.mkdir(args.output_path)
    except FileExistsError:
        pass

    strategies = []

    if args.segment_based_augmentation:
        strategies = [  # "swap_first_last",
            "remove_left_neighbor",
            "remove_right_neighbor",
            "remove_surrounding_neighbors",
            "label_wise_replacement",
            "shuffle_in_entity",
            "shuffle_in_segments"]
    if args.character_based_augmentation:
        strategies = ["reverse_letter_case",
                      "random_delete_character",
                      "shuffle_characters_in_token"]
    SAMPLE_RATIO = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1]
    N_ITERATION = 1

    with open(f"./{args.output_path}/augmentation_stats.tsv", "w") as file:
        file.write("strategy\tn_sentences_total\tn_entity_sentences\tn_samples\t"
                   "n_iteration\tn_augmentation\tsample_ratio\taugmentation_ratio\ttotal_ratio\n")

        for strategy, ratio in itertools.product(strategies, SAMPLE_RATIO):
            augmentation = Augmentation(input_path=args.input_path,
                                        word_column=args.word_column,
                                        tag_columns=args.tag_columns,
                                        main_entity_column=args.main_entity_column,
                                        sample_ratio=ratio,
                                        p_augmentation=args.p_augmentation,
                                        n_iteration=N_ITERATION,
                                        seed=args.seed
                                        )
            print(f"Create augmentation: \nStrategy: {strategy}\tSample Ratio: {ratio}\tN_iteration: {N_ITERATION}")
            if args.segment_based_augmentation:
                augmentation.word_based_augmentation(strategy=strategy)
            if args.character_based_augmentation:
                augmentation.character_based_augmentation(strategy=strategy)

            augmented_data = augmentation.augmentation_samples
            n_sentences, n_ent_sentences, n_samples, n_aug = augmentation.get_sizes()

            if args.output_path is not None:
                if args.to_tsv:
                    to_tsv(f"{args.output_path}/{strategy}-{ratio}-{N_ITERATION}.tsv", augmented_data)
                if args.to_json:
                    to_json(f"{args.output_path}/{strategy}-{ratio}-{N_ITERATION}.json", augmented_data,
                            args.json_columns)

            file.write(f"{strategy}\t{n_sentences}\t{n_ent_sentences}\t{n_samples}\t"
                       f"{N_ITERATION}\t{n_aug}\t{ratio}\t{n_aug / n_ent_sentences}\t{n_aug / n_sentences}\n"
                       )
