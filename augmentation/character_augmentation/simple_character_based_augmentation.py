from dataset import SequenceSegmentation
from typing import List

import numpy as np
import random
import math


class SimpleCharacterBasedAugmentation(SequenceSegmentation):
    def __init__(self, sequence: List[str], labels: List[str]):
        super().__init__(sequence=sequence, labels=labels)

    def random_reverse_letter_case(self, p: float = 0.5, seed: int = 0):
        """
        Randomly reversing the case of a letter. I.e.: lower -> upper  and upper -> lower
        :param p: Random probability
        :param seed: Random seed
        :return: List of tokens containing reversed letters cases
        """
        sequence = []
        for token in self.get_tokens_from_segments():
            # ignore punctuations
            # np.random.seed(seed)
            if np.random.binomial(1, p, 1)[0] == 1 and len(token) > 2:
                letter_id = math.floor(p * (len(token) - 1))  # select letter for case reversion
                letters = list(token)
                letters[letter_id] = letters[letter_id].upper() if letters[letter_id].islower() else letters[letter_id].lower()
                token = "".join(letters)
            sequence.append(token)
        return sequence

    def random_delete_character(self, p: float = 0.5, seed: int = 0):
        """
        Random delete a character / letter from token.
        :param p: Random probability
        :param seed: Random seed
        :return: List of tokens with omitted characters
        """
        sequence = []
        for token in self.get_tokens_from_segments():
            # Length of current token should be at least 2
            # np.random.seed(seed)
            if np.random.binomial(1, p, 1)[0] == 1 and len(token) > 1:
                char_id = math.floor(p * (len(token) - 1))
                chars = list(token)
                chars.pop(char_id)
                token = "".join(chars)
            sequence.append(token)
        return sequence

    def random_shuffle_chars_in_token(self, p: float = 0.5, seed: int = 0):
        """
        Random shuffle given token with a length of 2 characters or more.
        :param p: Random probability
        :param seed: Seed for randomisation
        :return: List of sequence with shuffled tokens
        """
        sequence = []
        for token in self.get_tokens_from_segments():
            # np.random.seed(seed)
            if np.random.binomial(1, p, 1)[0] == 1 and len(token) > 2:
                chars = list(token)
                random.seed(seed)
                random.shuffle(chars)
                token = "".join(chars)
            sequence.append(token)
        return sequence

    def get_tokens_from_segments(self):
        for segment in self.get_tags_based_segments():
            yield from segment


if __name__ == "__main__":
    tokens = ["My", "name", "is", "Monkey", "D.", "Luffy", ".", "I", "'", "m", "gonna", "be", "King", "of", "the", "Pirates"]
    tags = ["O", "O", "O", "B-name", "I-name", "I-name", "O", "O", "O", "O", "O", "O", "B-title", "I-title", "I-title", "I-title"]
    char_aug = SimpleCharacterBasedAugmentation(tokens, tags)
    print(char_aug.random_reverse_letter_case())
    print(char_aug.random_delete_character())
    print(char_aug.random_shuffle_chars_in_token())

    print()

