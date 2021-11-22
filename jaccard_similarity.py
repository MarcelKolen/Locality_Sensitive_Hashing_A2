import numpy as np
from scipy.sparse import csr_matrix

import time

from similarity_setup import SimilarityBase


class JaccardSimilarityBase(SimilarityBase):
    row_permutations = []
    signature_size = ...
    user_signatures = ...
    block_amount = ...
    block_column_size = ...
    buckets_amount = ...
    buckets = {}

    def __random_permutation(self, size):
        return np.random.choice(np.arange(0, size), replace=False, size=(size,))

    def __generate_random_permutations(self):
        shape = self.user_movie_matrix.get_shape()[1]

        # Generate random permutations to enable signature generation.
        for i in range(0, self.signature_size):
            self.row_permutations.append(self.__random_permutation(shape))

    ### OLD DEPRECIATED AND SLOW METHOD!
    # def __generate_signatures_for_users(self, user_range):
    #     column_range_max = self.user_movie_matrix.get_shape()[1]
    #
    #     # For all columns (movies) loop through a range of users
    #     # and calculate signatures.
    #     for c in range(0, column_range_max):
    #         for r in user_range:
    #             if self.user_movie_matrix[r, c] > 0:
    #                 for i, h in enumerate(self.row_permutations):
    #                     if np.isnan(self.user_signatures[r, i]) or h[c] < self.user_signatures[r, i]:
    #                         self.user_signatures[r, i] = h[c]

    def __generate_signatures_for_users(self, user_range):
        for r in user_range:
            non_zero_column_positions = np.nonzero(self.user_movie_matrix.getrow(r).toarray()[0])[0]
            for i, h in enumerate(self.row_permutations):
                self.user_signatures[r, i] = np.min(h[non_zero_column_positions])

    def __hash_blocks(self, user_range):
        for r in user_range:
            for block in range(0, self.block_amount):
                block_hash = hash(tuple(self.user_signatures[r][block*self.block_column_size:block*self.block_column_size+self.block_column_size]))
                if (bucket := self.buckets.get(block_hash, None)) is not None:
                    bucket.append(r)
                else:
                    self.buckets[block_hash] = [r]

    def __calculate_jaccard_similarity(self, usr_0, usr_1):
        usr_0_cp_binary = np.where(np.array(self.user_movie_matrix.getrow(usr_0).toarray()) > 0, 1, 0)
        usr_1_cp_binary = np.where(np.array(self.user_movie_matrix.getrow(usr_1).toarray()) > 0, 1, 0)

        return np.bitwise_and(usr_0_cp_binary, usr_1_cp_binary).sum() / np.bitwise_or(usr_0_cp_binary, usr_1_cp_binary).sum()

    def __find_similar_users(self):
        for bucket in self.buckets.items():
            bucket = bucket[1]
            bucket_len = len(bucket)

            if bucket_len > 1:

                for i in range(0, bucket_len):
                    for j in range(i + 1, bucket_len):
                        if (js := self.__calculate_jaccard_similarity(bucket[i], bucket[j])) > self.similarity_limit:
                            self.similarity_output_function(bucket[i], bucket[j])
                            print(f"Pair {bucket[i]}:{bucket[j]} are similar ({js})")

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

        self.__generate_random_permutations()

    def __call__(self, *args, **kwargs):
        print("Now running the Jaccard Similarity Routine")

        print("Generating signatures")
        start_time = time.perf_counter()
        self.__generate_signatures_for_users(range(0, self.user_movie_matrix_shape[0]))
        print(f"Generating signatures took: {time.perf_counter() - start_time}\n")

        print("Generating hashes")
        start_time = time.perf_counter()
        self.__hash_blocks(range(0, self.user_movie_matrix_shape[0]))
        print(f"Generating hashes took: {time.perf_counter() - start_time}\n")

        print("Evaluating similarity canidates")
        start_time = time.perf_counter()
        self.__find_similar_users()
        print(f"Evaluating similarity canidates took: {time.perf_counter() - start_time}\n")


        return
