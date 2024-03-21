import math

from dataset.segmentation import SequenceSegmentation

from copy import deepcopy
from typing import List, Dict

import numpy as np
import itertools
import random


class SimpleSegmentBasedAugmentation(SequenceSegmentation):
    def __init__(self, sequence: List[str], labels: List[str]):
        super().__init__(sequence=sequence, labels=labels)

    def random_swap_first_last_segment_tokens(self, p: float = 0.5):
        """
        Randomly swap first and last token inside entity segment
        :param p: Random probability
        :return: List of sequences with randomly tokens swapped entity
        """
        sequence = deepcopy(self.sequence)
        for segment, positions in self.get_annotated_segment():
            if len(segment) >= 1:
                seed = math.floor(len(segment) * p)
                np.random.seed(seed)
                if np.random.binomial(1, p, 1)[0] == 1:
                    sequence[positions[0]] = segment[-1]
                    sequence[positions[-1]] = segment[0]
        return sequence

    def random_remove_entity_neighbor(self,
                                      left: bool = True,
                                      right: bool = False,
                                      p: float = 0.5,
                                      return_pos_ids: bool = False):
        """
        Randomly remove neighbor tokens of entity spans.
        :param left: Whether the left neighbored token should be deleted
        :param right: Whether right neighbored token should be deleted
        :param p: Random probability
        :param return_pos_ids: Whether a list of tokens positions within segment should be returned
        :return: Modified list of tokens, labels and position ids
        """
        sequence = deepcopy(self.sequence)
        labels = deepcopy(self.labels)
        pos_ids = []    # Positions of neighbor tokens
        for segment, positions in self.get_annotated_segment():
            if left:
                pos_ids.append(positions[0] - 1)
            if right:
                pos_ids.append(positions[-1] + 1)

        # Random seed for reproducibility using length of pos_ids and probability value p
        seed = math.floor(len(pos_ids) * p)
        np.random.seed(seed)
        random_distribution = np.random.binomial(1, p, len(pos_ids)).tolist() if not left or not right else [1] * len(
            pos_ids)

        pos_ids = [pos_ids[i] for i, rand in enumerate(random_distribution)
                   if rand != 0  # Value from random binomial distribution shouldn't be 0
                   # Entity token shouldn't be the first or the last token of the sequence
                   and 0 <= pos_ids[i] < len(sequence)
                   # Ignore strategy if the token of prev or next segment is also an entity
                   and ("B-" not in labels[pos_ids[i]] and
                        "I-" not in labels[pos_ids[i]]
                        )
                   ]
        # Generate augmented tokens sequence and adjust corresponding labels list
        sequence = [sequence[j] for j, _ in enumerate(sequence) if j not in pos_ids]
        labels = [labels[j] for j, _ in enumerate(labels) if j not in pos_ids]
        if not return_pos_ids:
            return sequence, labels
        else:
            return sequence, labels, pos_ids

    def label_wise_token_replacement(self, labels_to_tokens_map: Dict[str, List[str]], p: float = 0.5):
        """
        Randomly replace an entity token with another in the same entity-tag category
        :param labels_to_tokens_map: Dictionary of labels tokens mapping
        :param p: Random probability
        :return: List of tokens sequence with replaced entity tokens
        """
        sequence = deepcopy(self.sequence)
        labels = self.labels
        chosen = []
        for segment, positions in self.get_annotated_segment():
            for i, pos in enumerate(positions):
                chosen.append(segment[i])
                # Create seed for reproducibility based on length of segment and p
                seed = math.floor(len(segment) * p)
                np.random.seed(seed)
                if np.random.binomial(1, p, 1)[0] == 1:
                    # Random select a token from the same label class of the original token
                    replacement = np.random.choice(labels_to_tokens_map[labels[pos]])
                    if replacement not in chosen:
                        segment[i] = replacement
                        chosen.append(replacement)
                sequence[pos] = segment[i]
        return sequence

    def shuffle_within_entity_segment(self, p: float = 0.5):
        """
        Shuffle random annotated entity segment
        :param p: Random probability
        :return: List of tokens sequence with shuffled entity spans
        """
        sequence = deepcopy(self.sequence)
        for segment, positions in self.get_annotated_segment():
            # Random seed for reproducibility
            seed = math.floor(len(segment) * p)
            np.random.seed(seed)
            if len(segment) > 1 and np.random.binomial(1, p, 1)[0] == 1:
                np.random.shuffle(segment)
            for i, _ in enumerate(sequence):
                if i in positions:
                    sequence[i] = segment[positions.index(i)]
        return sequence

    def shuffle_within_segments(self, p: float = 0.5):
        """
        Random shuffle segments within a sequence
        :param p: Random probability
        :return: List of randomly shuffled tokens segments
        """
        segmented = self.get_tags_based_segments()
        for i, _ in enumerate(segmented):
            # Random seed for reproducibility
            seed = math.floor(len(segmented[i]) * p)
            np.random.seed(seed)
            if len(segmented[i]) > 1 and np.random.binomial(1, p, 1)[0] == 1:
                np.random.shuffle(segmented[i])
        sequence = list(itertools.chain.from_iterable(segmented))
        return sequence


if __name__ == "__main__":
    np.random.seed(1)
    print(np.random.binomial(1, 0.5, 1)[0])

    tokens = ["Hello", ",", "my", "name", "is", "Monkey", "D.", "Luffy", "and",
              "I", "am", "gonna", "be", "the", "King", "of", "the", "Pirates", "."]

    tags_0 = ["O", "O", "O", "O", "O", "B-anon", "I-anon", "I-anon", "O",
              "O", "O", "O", "O", "O", "B-anon", "I-anon", "I-anon", "I-anon", "O"]

    tags_1 = ["O", "O", "O", "O", "O", "B-name", "I-name", "I-name", "O",
              "O", "O", "O", "O", "O", "B-title", "I-title", "I-title", "I-title", "O"]

    aug = SimpleSegmentBasedAugmentation(tokens, tags_0)
    print(aug.get_tags_based_segments())

    assert len(tokens) == len(tags_0), f"{len(tokens)}\t{len(tags_0)}"
    assert len(tokens) == len(tags_1), f"{len(tokens)}\t{len(tags_1)}"

    # segmentation = SequenceSegmentation(tokens, labels=tags_0)
    # for t, p in segmentation.get_annotated_segment():
    #    print(t, p)

    # Test adding noises into sequence
    augmentation = SimpleSegmentBasedAugmentation(tokens, tags_0)

    swapped_entity_tokens = augmentation.random_swap_first_last_segment_tokens(p=0.5)
    print(f"Random Swap: {swapped_entity_tokens}\n")

    shuffled_annotated_segment = augmentation.shuffle_within_entity_segment(p=0.5)
    print(f"Random shuffle annotation: {shuffled_annotated_segment}\n")

    omitted_prev_neighbor = augmentation.random_remove_entity_neighbor(p=0.5)
    print(f"Random Omit left neighbor: {omitted_prev_neighbor}\n")

    omitted_following_neighbor = augmentation.random_remove_entity_neighbor(left=False, right=True, p=0.5)
    print(f"Random Omit right neighbor: {omitted_following_neighbor}\n")

    omitted_surrounding_neighbors = augmentation.random_remove_entity_neighbor(left=True, right=True, p=0.5)
    print(f"Random Omit surrounding neighbors: {omitted_surrounding_neighbors}\n")

    sis_sequence = augmentation.shuffle_within_segments(p=0.5)
    print(f"Random shuffle in segments: {sis_sequence}\n")
    print()
