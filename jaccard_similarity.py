### Advances In Data Mining
### Assignment 2
### Luit Verschuur 1811053, Marcel Kolenbrander 1653415
import multiprocessing

import numpy as np

from similarity_setup import SimilarityBase


class JaccardSimilarity(SimilarityBase):
    column_permutations = ...
    binary_matrix = []

    def random_permutation(self, size):
        """
        Return an array with a random arrangement between 0 and size with no replacements (no duplicates).

        :param size: Number of elements and the indication of the largest element (size - 1) in the random array.
        :return: A random array of with the length of size with values between 0 and size - 1.
        """

        return np.random.choice(np.arange(0, size), replace=False, size=(size,))

    def generate_random_permutations(self):
        """
        Generate a set of random permutation arrays using the __random_permutation function.
        The size of this random random permutation array is the user signature size and the number of columns
        in the data matrix.

        :return:
        """

        shape = self.user_movie_matrix_shape[1]

        # Generate random permutations to enable signature generation.
        for i in range(0, self.signature_size):
            self.column_permutations[i] = self.random_permutation(shape)

    def calculate_user_similarity(self, usr_0, usr_1):
        """
        Calculate the jaccard similarity (a value between 0. and 1.) of two given users. Compare the users using their
        original data entries. The jaccard similarity is the quotient between the intersect and the union of the users.

        :param usr_0: A users ID
        :param usr_1: A users ID
        :return: The jaccard similarity between 0. and 1. (inclusive boundaries)
        """

        # Convert user movie ratings to binary values. If given a rating (user rated movie) give a 1,
        # else a 0 (user has not rated movie)

        usr_0_binary = self.binary_matrix.getrow(usr_0).toarray()
        usr_1_binary = self.binary_matrix.getrow(usr_1).toarray()

        # The intersect can also be seen as the "and" and the union can also been seen as an "or". Calculate the jaccard
        # similarity by taking the quotient of the sums of the and, and or of the two users.
        return np.bitwise_and(usr_0_binary, usr_1_binary).sum() / np.bitwise_or(usr_0_binary, usr_1_binary).sum()

    def generate_signatures_for_users(self, user_range):
        """
        For each user (row) in a given range, calculate the min hashes using a random permutation of the columns.
        The collection of min hashes form the signature for the users.

        :param user_range: A range of users (rows) over which the loop should run
        :return: A 2D array containing signatures for the users based on minhashing
        """

        # Init empty user signatures of size given by the user_range and the signature size.
        user_signatures = np.empty(shape=(user_range.stop - user_range.start, self.signature_size))
        user_signatures[:] = np.NaN

        # For each users (row) calculate the minhash.
        for r in user_range:
            # Find all non zero column positions and store them in an array to be used as a mask.
            non_zero_column_positions = self.user_movie_matrix.getrow(r).nonzero()[1]

            # For each random permutation, find the corresponding signature element by taking the permutation value for
            # the first column to corresponding with a non-zero value. This can be done by applying a mask over the
            # random permutations and finding the lowest value.
            for i, h in enumerate(self.column_permutations):
                user_signatures[r - user_range.stop, i] = np.min(h[non_zero_column_positions])

        return user_signatures

    def init(self, *args, **kwargs):
        self.column_permutations = np.empty((self.signature_size, self.user_movie_matrix_shape[1]))

        self.generate_random_permutations()

        # Convert matrix into a binary matrix where all 0 values remain zero and all non zero values are 1.
        self.binary_matrix = self.user_movie_matrix.copy()
        self.binary_matrix.data = np.ones_like(self.binary_matrix.data)

        self.welcome_text = "Now running the Jaccard Similarity Routine"
