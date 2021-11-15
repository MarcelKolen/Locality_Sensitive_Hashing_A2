import numpy as np
from scipy.sparse import csr_matrix

from similarity_setup import SimilarityBase


class JaccardSimilarityBase(SimilarityBase):
    def __random_row_permutation(self, size_of_row):
        return np.random.choice(np.arange(0, size_of_row), replace=False, size=(size_of_row, ))

    def __call__(self, *args, **kwargs):
        print("Now running the Jaccard Similarity Routine")

        # Print found pair to file (File path is pre-set)
        self.similarity_output_function(0, 1)

        # Hier moet de routine komen voor de JS methode.

        return
