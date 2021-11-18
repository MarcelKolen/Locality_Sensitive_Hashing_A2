import numpy as np
from scipy.sparse import csr_matrix

from similarity_setup import SimilarityBase


class JaccardSimilarityBase(SimilarityBase):
    row_permutations = []
    signature_size = ...
    user_signatures = ...
    block_amount = ...
    block_column_size = ...
    buckets_amount = ...
    buckets = []

    def __random_permutation(self, size):
        return np.random.choice(np.arange(0, size), replace=False, size=(size,))

    def __generate_random_permutations(self):
        shape = self.user_movie_matrix.get_shape()[1]

        # Generate random permutations to enable signature generation.
        for i in range(0, self.signature_size):
            self.row_permutations.append(self.__random_permutation(shape))

    def __generate_signatures_for_users(self, user_range):
        column_range_max = self.user_movie_matrix.get_shape()[1]

        # For all columns (movies) loop through a range of users
        # and calculate signatures.
        for r in range(user_range[0], user_range[1]):
            for c in range(0, column_range_max):
                if self.user_movie_matrix[r,c] > 0:
                    for i, h in enumerate(self.row_permutations):
                        if np.isnan(self.user_signatures[r, i]) or h[c] < self.user_signatures[r, i]:
                            self.user_signatures[r, i] = h[c]

    def __hash_blocks(self, user_range):
        for r in range(user_range[0], user_range[1]):
            for block in range(0, self.block_amount):
                self.buckets[abs(hash(tuple(self.user_signatures[r][block*self.block_column_size:block*self.block_column_size+self.block_column_size]))) % self.buckets_amount].append(r)

    def __calculate_jaccard_similarity(self, usr_0, usr_1):
        usr_0_cp_binary = np.where(np.array(self.user_movie_matrix.getrow(usr_0).toarray()) > 0, 1, 0)
        usr_1_cp_binary = np.where(np.array(self.user_movie_matrix.getrow(usr_1).toarray()) > 0, 1, 0)

        return np.bitwise_and(usr_0_cp_binary, usr_1_cp_binary).sum() / np.bitwise_or(usr_0_cp_binary, usr_1_cp_binary).sum()


    def __find_similar_users(self):
        for bucket in self.buckets:
            u_bucket = list(set(bucket))
            u_bucket_len = len(u_bucket)

            if u_bucket_len < 2:
                continue

            for i in range(0, u_bucket_len):
                for j in range(i + 1, u_bucket_len):
                   if self.__calculate_jaccard_similarity(u_bucket[i], u_bucket[j]) > self.similarity_limit:
                        self.similarity_output_function(i, j)
                        print(f"Pair {u_bucket[i]}:{u_bucket[j]} are similar")

    def __init__(self, signature_size_in=None, block_amount_in=None, block_row_size_in=None, buckets_amount_in=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if signature_size_in is not None:
            self.signature_size = signature_size_in
        else:
            raise ValueError("Signature size not initiated")

        if block_amount_in is not None:
            self.block_amount = block_amount_in
        else:
            raise ValueError("Band amount not initiated")

        if block_row_size_in is not None:
            self.block_column_size = block_row_size_in
        else:
            raise ValueError("Band row size not initiated")

        if buckets_amount_in is not None:
            self.buckets_amount = buckets_amount_in
        else:
            raise ValueError("Band row size not initiated")

        if self.block_column_size * self.block_amount != self.signature_size:
            raise ValueError("Band amount and Band size do not match the signature size")

        self.user_signatures = np.empty(shape=(self.user_movie_matrix_shape[0], self.signature_size))
        self.user_signatures[:] = np.NaN

        self.buckets = [[] for i in range(self.buckets_amount)]

        self.__generate_random_permutations()

    def __call__(self, *args, **kwargs):
        print("Now running the Jaccard Similarity Routine")

        # print(self.user_movie_matrix.get_shape()[0])

        print("Generating signatures")
        self.__generate_signatures_for_users((0, 500))
        print("Generating hashes")
        self.__hash_blocks((0, 500))
        print("Finding similars")
        self.__find_similar_users()

        print(self.buckets)

        # Print found pair to file (File path is pre-set)
        # self.similarity_output_function(0, 1)

        return
