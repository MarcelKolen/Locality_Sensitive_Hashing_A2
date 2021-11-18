import numpy as np
from scipy.sparse import csr_matrix

from similarity_setup import SimilarityBase


class JaccardSimilarityBase(SimilarityBase):
    row_permutations = []
    signature_size = ...
    user_signatures = ...

    def __random_permutation(self, size):
        return np.random.choice(np.arange(0, size), replace=False, size=(size,))

    def __generate_random_permutations(self):
        shape = self.user_movie_matrix.get_shape()[1]

        # Generate random permutations to enable signature generation.
        for i in range(0, self.signature_size):
            self.row_permutations.append(self.__random_permutation(shape))

    def __generate_signatures_for_users(self, user_range):
        column_range_max = self.user_movie_matrix.get_shape()[1]

        dense_matrix = self.user_movie_matrix.todense()

        # For all columns (movies) loop through a range of users
        # and calculate signatures.
        for c in range(0, column_range_max):
            for r in range(user_range[0], user_range[1]):
                if dense_matrix[r, c] > 0:
                    for i, h in enumerate(self.row_permutations):
                        if np.isnan(self.user_signatures[r, i]) or h[c] < self.user_signatures[r, i]:
                            self.user_signatures[r, i] = h[c]

    def __init__(self, signature_size_in=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if signature_size_in is not None:
            self.signature_size = signature_size_in
        else:
            raise ValueError("Signature size not initiated")

        self.user_signatures = np.empty(shape=(self.user_movie_matrix.get_shape()[0], self.signature_size))
        self.user_signatures[:] = np.NaN

        self.__generate_random_permutations()

    def __call__(self, *args, **kwargs):
        print("Now running the Jaccard Similarity Routine")

        # print(self.user_movie_matrix.get_shape()[0])

        self.__generate_signatures_for_users((0, 20))

        print(self.user_signatures)
        # Hier moet de routine komen voor de JS methode.

        # Print found pair to file (File path is pre-set)
        # self.similarity_output_function(0, 1)

        return
