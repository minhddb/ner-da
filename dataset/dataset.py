import numpy as np
import itertools


class Dataset:
    def __init__(self, inp_path: str, words_col: int, *tags_col: int):
        self.inp_path = inp_path
        self.words_col = words_col
        self.tags_col = tags_col
        with open(inp_path, "r", encoding="utf-8") as inp_f:
            self.lines = inp_f.readlines()

    def __len__(self):
        """ Return number of entity sequences in corpus"""
        return len(self.read_tsv_to_list())

    def __call__(self):
        """ Return list of entity sequences"""
        return [entity_seq for entity_seq in self.get_entity_sequence()]

    def get_entity_sequence(self):
        """
        Yield sequences with annotated entities only
        """
        for _, sequence in enumerate(self.read_tsv_to_list()):
            if any("B-" in label for label in sequence[1]):
                yield sequence

    def read_tsv_to_list(self):
        corpus = []
        sequence = []
        for _, line in enumerate(self.lines):
            # TODO: Revise this part, since it's not suitable for other corpora
            if not line.startswith("# newdoc id") and not line.startswith("# sent_id"):
                current_line = line.strip()
                if current_line != "":
                    split_line = current_line.split()
                    sequence.append(list(itertools.chain.from_iterable([[split_line[self.words_col]],
                                                                        [split_line[i] for i in self.tags_col]])))
                else:
                    sequence = np.transpose(np.array(sequence)).tolist()
                    corpus.append(sequence)
                    sequence = []
        return corpus
