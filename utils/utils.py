from typing import List
import numpy as np
import json


def to_tsv(output_path: str, sequences: List[List[str]]):
    """
    Write augmented sequences to tsv
    :param output_path: Output path
    :param sequences: List of lists of tokens and tags
    """
    if output_path:
        with open(output_path, "w", encoding="utf-8") as out_f:
            for i, sequence in enumerate(sequences):
                try:
                    transposed_sequence = list(np.transpose(np.array(sequence)))
                except ValueError:
                    print(i)
                    for lst in sequence:
                        print(lst, len(lst), sep="\t")
                    continue
                for ind, _ in enumerate(transposed_sequence):
                    out_f.write("\t".join(transposed_sequence[ind]) + "\n")
                    if ind == len(transposed_sequence) - 1:
                        out_f.write("\n")


def to_json(output_path: str, sequences: List[List[str]], columns: List[str] = None):
    """
    Write augmented sequences to JSON file
    :param output_path: Path to output file
    :param sequences: List of lists of tokens and tags
    :param columns: List of keys names
    """
    if output_path:
        with open(output_path, "w", encoding="utf-8") as out_json:
            for i, _ in enumerate(sequences):
                json_dict = {}
                for j, sequence in enumerate(sequences[i]):
                    if columns is not None:
                        assert len(columns) == len(sequences[i])
                        json_dict.update({columns[j]: sequence})
                    else:
                        json_dict.update({j: sequence})
                json.dump(json_dict, out_json)
                out_json.write("\n")
