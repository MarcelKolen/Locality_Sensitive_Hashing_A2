import numpy as np
from scipy.sparse import csr_matrix

from similarity_setup import SimilarityBase


class DiscreteCosineSimilarityBase(SimilarityBase):
    def __call__(self, *args, **kwargs):
        print("Now running the Discrete Cosine Similarity Routine")

        # Print found pair to file (File path is pre-set)
        self.similarity_output_function(0, 1)

        # Hier moet de routine komen voor de DCS methode.

        return
