from .dataset import Dataset

from typing import List
import spacy


class Mappings:
    # TODO: Finish commenting this part
    def __init__(self, inp_dataset: List[List[str]], spacy_model: str = "de_core_news_md"):
        self.inp_dataset = inp_dataset
        self.model = spacy.load(spacy_model)

    def map_entity_to_distribution(self):
        pass
        # TODO: Map entities to their distribution in corpus (ignore 'O')

    def map_labels_to_tokens(self, entity_column: int = 1):
        """
        Create labels to tokens mapping. This mapping will be utilised for label-wise and similarity-wise replacements.
        :return:
        """
        labels_to_tokens_mappings = {}
        for anno in self.inp_dataset:
            for seq, tag in zip(anno[0], anno[entity_column]):
                if tag not in labels_to_tokens_mappings.keys():
                    labels_to_tokens_mappings.update({tag: [seq]})
                else:
                    if seq not in labels_to_tokens_mappings[tag]:
                        labels_to_tokens_mappings[tag].append(seq)
        return labels_to_tokens_mappings

    def map_tokens_to_similarity_scores(self, inp_word: str, vocabs_list: List[str]):
        """

        :param inp_word:
        :param vocabs_list:
        :return:
        """
        inp_vocab = self.model(inp_word)
        # Create a list of spacy vocabularies with valid word vector( i.e., word vector is not empty)
        spacy_vocabularies = [self.model(word) for word in vocabs_list if self.model(word).vector_norm]

        # Compute similarity between inp word and tokens from vocabs list using out-of-the-shelf spaCy model
        tokens_to_similarities_mapping = {}
        for vocab in spacy_vocabularies:
            tokens_to_similarities_mapping.update({str(vocab): inp_vocab.similarity(vocab)})

        # Sort mapping by similarity scores
        sorted_mappings = {vocabulary: sim for vocabulary, sim in sorted(
            tokens_to_similarities_mapping.items(),
            key=lambda sim: sim[1],
            reverse=True
        )
                           }
        return sorted_mappings

    def get_top_n_similarities(self, inp_word: str, vocabs_list: List[str], n: int = 5):
        """

        :param inp_word:
        :param vocabs_list:
        :param n:
        :return:
        """
        list_of_top_similarities = list(self.map_tokens_to_similarity_scores(inp_word, vocabs_list=vocabs_list).keys())
        # Remove word from list of top similarities to avoid duplication
        try:
            list_of_top_similarities.remove(inp_word)
        except ValueError:
            return list_of_top_similarities[:n]
        return list_of_top_similarities[:n]


if __name__ == "__main__":
    PATH = ""

    dataset = Dataset(PATH, 0, 2)
    read_dataset = dataset.read_tsv_to_list()  # dataset.read_tsv_to_list()
    entity_sequence = dataset()

    mappings = Mappings(read_dataset)

    labels_map = mappings.map_labels_to_tokens()

    relevant_tokens = labels_map["B-nat-name"]

    token = "Hanami"
    # tokens_to_sim_scores_mappings = mappings.map_tokens_to_similarity_scores(token, relevant_tokens)
    top_n_similarities = mappings.get_top_n_similarities(token, relevant_tokens, n=6)
    print(top_n_similarities)
    print()
