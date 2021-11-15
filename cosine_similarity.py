import numpy as np
from scipy.sparse import csr_matrix

from similarity_setup import SimilarityBase


class CosineSimilarityBase(SimilarityBase):
    def __call__(self, *args, **kwargs):
        print("Now running the Cosine Similarity Routine")

        # Print found pair to file (File path is pre-set)
        self.similarity_output_function(0, 1)

        # Hier moet de routine komen voor de CS methode.

        return
