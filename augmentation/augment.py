import math
import random
from typing import Set

from dataset import Dataset, Mappings
from augmentation.character_augmentation import SimpleCharacterBasedAugmentation
from augmentation.segment_augmentation import SimpleSegmentBasedAugmentation


class Augmentation:
    """
    Generate augmented data using different augmentation strategies
    """

    def __init__(self,
                 input_path: str,
                 word_column: int,
                 tag_columns: Set[int] | int,
                 main_entity_column: int = None,
                 sample_ratio: float = 0.1,
                 p_augmentation: float = 0.5,
                 n_iteration: int = 1,
                 seed: int = 42,
                 ):
        """
        :param input_path: Path to input file
        :param word_column: Index for word column
        :param tag_columns: Indices for tag columns
        :param main_entity_column: Index of main entity column for multi-columns tagging. Default: 1
        :param sample_ratio: Ratio to retrieve sample amount of annotated entity sequences
        :param p_augmentation: Probability to randomly decide whether the given segment should be augmented
        :param n_iteration: Number of augmentation rounds
        :param seed: Random seed
        """
        self.tag_columns = {tag_columns} if int == type(tag_columns) else tag_columns
        self.main_entity_column = 1 if main_entity_column is None else main_entity_column
        self.dataset = Dataset(input_path, word_column, *self.tag_columns)
        self.all_sequences = self.dataset.read_tsv_to_list()
        self.entity_sequences = [sequence for sequence in self.dataset.get_entity_sequence()]
        self.labels_to_tokens_mapping = Mappings(self.all_sequences).map_labels_to_tokens(self.main_entity_column)
        self.sample_ratio = sample_ratio
        self.p_augmentation = p_augmentation
        self.n_iteration = n_iteration
        self.n_samples = math.floor(self.sample_ratio * len(self.entity_sequences))
        self.seed = seed
        self.augmentation_samples = []

    def word_based_augmentation(self, strategy: str = "swap_first_last"):
        """
        :param strategy: Augmentation strategy
        """
        for sample in self.get_samples():
            augment = SimpleSegmentBasedAugmentation(sequence=sample[0], labels=sample[self.main_entity_column])
            already_exists = [sample[0]]
            current_iteration = 0  # Determine current augmentation round
            while current_iteration < self.n_iteration:
                if strategy == "swap_first_last":
                    augmented = augment.random_swap_first_last_segment_tokens(p=self.p_augmentation)
                    if augmented not in already_exists:
                        already_exists.append(augmented)
                        self.augmentation_samples.append([augmented] + sample[1:])

                if strategy == "remove_left_neighbor":
                    tmp_augmentation = []
                    augmented, labels, pos_ids = augment.random_remove_entity_neighbor(p=self.p_augmentation,
                                                                                       return_pos_ids=True
                                                                                       )
                    if augmented not in already_exists:
                        already_exists.append(augmented)
                        if len(list(self.tag_columns)) > 1:
                            tmp_augmentation.extend([augmented, labels])
                            tags = []  # Adjust tags for other columns if multi-columns
                            for i in range(1, len(sample)):
                                if i != self.main_entity_column:
                                    for j, _ in enumerate(sample[i]):
                                        if j not in pos_ids:
                                            tags.append(sample[i][j])
                                    tmp_augmentation.insert(i, tags)
                                    tags = []
                            self.augmentation_samples.append(tmp_augmentation)
                        else:
                            self.augmentation_samples.append([augmented, labels])

                if strategy == "remove_right_neighbor":
                    tmp_augmentation = []
                    augmented, labels, pos_ids = augment.random_remove_entity_neighbor(left=False,
                                                                                       right=True,
                                                                                       p=self.p_augmentation,
                                                                                       return_pos_ids=True
                                                                                       )
                    if augmented not in already_exists:
                        already_exists.append(augmented)
                        if len(list(self.tag_columns)) > 1:
                            tmp_augmentation.extend([augmented, labels])
                            tags = []
                            for i in range(1, len(sample)):
                                if i != self.main_entity_column:
                                    for j, _ in enumerate(sample[1]):
                                        if j not in pos_ids:
                                            tags.append(sample[i][j])
                                    tmp_augmentation.insert(i, tags)
                                    tags = []
                            self.augmentation_samples.append(tmp_augmentation)
                        else:
                            self.augmentation_samples.append([augmented, labels])

                if strategy == "remove_surrounding_neighbors":
                    tmp_augmentation = []
                    perturbed, labels, pos_ids = augment.random_remove_entity_neighbor(p=self.p_augmentation,
                                                                                       left=True,
                                                                                       right=True,
                                                                                       return_pos_ids=True)
                    if perturbed not in already_exists:
                        already_exists.append(perturbed)
                        if len(list(self.tag_columns)) > 1:
                            tmp_augmentation.extend([perturbed, labels])
                            tags = []  # Adjust tags for other columns if multi-columns
                            for i in range(1, len(sample)):
                                if i != self.main_entity_column:
                                    for j, _ in enumerate(sample[i]):
                                        if j not in pos_ids:
                                            tags.append(sample[i][j])
                                    tmp_augmentation.insert(i, tags)
                                    tags = []
                            self.augmentation_samples.append(tmp_augmentation)
                        else:
                            self.augmentation_samples.extend([perturbed, labels])

                if strategy == "label_wise_replacement":
                    augmented = augment.label_wise_token_replacement(
                        labels_to_tokens_map=self.labels_to_tokens_mapping,
                        p=self.p_augmentation
                    )
                    if augmented not in already_exists:
                        already_exists.append(augmented)
                        self.augmentation_samples.append([augmented] + sample[1:])

                if strategy == "shuffle_in_entity":
                    augmented = augment.shuffle_within_entity_segment(p=self.p_augmentation)
                    if augmented not in already_exists:
                        already_exists.append(augmented)
                        self.augmentation_samples.append([augmented] + sample[1:])

                if strategy == "shuffle_in_segments":
                    augmented = augment.shuffle_within_segments(p=self.p_augmentation)
                    if augmented not in already_exists:
                        already_exists.append(augmented)
                        self.augmentation_samples.append([augmented] + sample[1:])
                current_iteration += 1

    def character_based_augmentation(self, strategy: str = "reverse_letter_case"):
        """
        :param strategy: Augmentation strategy
        """
        for sample in self.get_samples():
            augment = SimpleCharacterBasedAugmentation(sequence=sample[0], labels=sample[1])
            already_exists = sample[0]
            current_iteration = 0
            while current_iteration < self.n_iteration:
                if strategy == "reverse_letter_case":
                    augmented = augment.random_reverse_letter_case(p=self.p_augmentation)
                    if augmented not in already_exists:
                        already_exists.append(augmented)
                        self.augmentation_samples.append([augmented] + sample[1:])

                if strategy == "delete_character":
                    augmented = augment.random_delete_character(p=self.p_augmentation)
                    if augmented not in already_exists:
                        already_exists.append(augmented)
                        self.augmentation_samples.append([augmented] + sample[1:])

                if strategy == "shuffle_characters_in_token":
                    augmented = augment.random_shuffle_chars_in_token(p=self.p_augmentation)
                    if augmented not in already_exists:
                        already_exists.append(augmented)
                        self.augmentation_samples.append([augmented] + sample[1:])
                current_iteration += 1

    def get_sizes(self):
        """
        :return: N sentences in dataset, N entity sentences, N samples and N augmented instances
        """
        return len(self.dataset), len(self.entity_sequences), self.n_samples, len(self.augmentation_samples)

    def get_samples(self):
        """
        Random sample and generate N annotated sentences from dataset based on sample ratio
        """
        random.seed(self.seed)
        yield from random.sample(self.entity_sequences, k=self.n_samples)
