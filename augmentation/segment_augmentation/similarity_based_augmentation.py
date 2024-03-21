from typing import List

from dataset import Mappings
from dataset import Dataset
from dataset import SequenceSegmentation


class SimilarityTokenAugmentation(SequenceSegmentation):
    def __init__(self, sequence: List[str], labels: List[str]):
        super().__init__(sequence=sequence, labels=labels)


    def similarity_context_token_replacement(self):
        pass

    def similarity_entity_token_replacement(self):
        pass

    def similarity_based_token_replacement(self):
        """

        :return:
        """
        # TODO: Replace token by top n similarity. Augment for n times
        pass
