### Advances In Data Mining
### Assignment 2
### Luit Verschuur 1811053, Marcel Kolenbrander 1653415


import numpy as np
from scipy.sparse import csr_matrix

import time

from similarity_setup import SimilarityBase


class JaccardSimilarityBase(SimilarityBase):
    column_permutations = []
    signature_size = ...
    user_signatures = ...
    block_amount = ...
    block_column_size = ...
    buckets = {}

    def __random_permutation(self, size):
        """
        Return an array with a random arrangement between 0 and size with no replacements (no duplicates).

        :param size: Number of elements and the indication of the largest element (size - 1) in the random array.
        :return: A random array of with the length of size with values between 0 and size - 1.
        """

        return np.random.choice(np.arange(0, size), replace=False, size=(size,))

    def __generate_random_permutations(self):
        """
        Generate a set of random permutation arrays using the __random_permutation function.
        The size of this random random permutation array is the user signature size and the number of columns
        in the data matrix.

        :return:
        """

        shape = self.user_movie_matrix.get_shape()[1]

        # Generate random permutations to enable signature generation.
        for i in range(0, self.signature_size):
            self.column_permutations.append(self.__random_permutation(shape))

    def __generate_signatures_for_users(self, user_range):
        """
        For each user (row) in a given range, calculate the min hashes using a random permutation of the columns.
        The collection of min hashes form the signature for the users.

        :param user_range: A range of users (rows) over which the loop should run
        :return:
        """

        # For each users (row) calculate the minhash.
        for r in user_range:
            # Find all non zero column positions and store them in an array to be used as a mask.
            non_zero_column_positions = np.nonzero(self.user_movie_matrix.getrow(r).toarray()[0])[0]

            # For each random permutation, find the corresponding signature element by taking the permutation value for
            # the first column to corresponding with a non-zero value. This can be done by applying a mask over the
            # random permutations and finding the lowest value.
            for i, h in enumerate(self.column_permutations):
                self.user_signatures[r, i] = np.min(h[non_zero_column_positions])

    def __calculate_jaccard_similarity(self, usr_0, usr_1):
        """
        Calculate the jaccard similarity (a value between 0. and 1.) of two given users. Compare the users using their
        original data entries. The jaccard similarity is the quotient between the intersect and the union of the users.

        :param usr_0: A users ID
        :param usr_1: A users ID
        :return: The jaccard similarity between 0. and 1. (inclusive boundaries)
        """

        # Convert user movie ratings to binary values. If given a rating (user rated movie) give a 1,
        # else a 0 (user has not rated movie)
        usr_0_cp_binary = np.where(np.array(self.user_movie_matrix.getrow(usr_0).toarray()) > 0, 1, 0)
        usr_1_cp_binary = np.where(np.array(self.user_movie_matrix.getrow(usr_1).toarray()) > 0, 1, 0)

        # The intersect can also be seen as the "and" and the union can also been seen as an "or". Calculate the jaccard
        # similarity by taking the quotient of the sums of the and and or of the two users.
        return np.bitwise_and(usr_0_cp_binary, usr_1_cp_binary).sum() / np.bitwise_or(usr_0_cp_binary, usr_1_cp_binary).sum()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__generate_random_permutations()

    def __call__(self, *args, **kwargs):
        print("Now running the Jaccard Similarity Routine")

        print("Generating signatures")
        start_time = time.perf_counter()
        self.__generate_signatures_for_users(range(0, int(self.user_movie_matrix_shape[0]/4)))
        print(f"Generating signatures took: {time.perf_counter() - start_time}\n")

        print("Generating hashes")
        start_time = time.perf_counter()
        self.__hash_blocks(range(0, int(self.user_movie_matrix_shape[0]/4)))
        print(f"Generating hashes took: {time.perf_counter() - start_time}\n")

        print("Evaluating similarity canidates")
        start_time = time.perf_counter()
        self.__find_similar_users(self.__calculate_jaccard_similarity)
        print(f"Evaluating similarity canidates took: {time.perf_counter() - start_time}\n")

        return
