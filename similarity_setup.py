### Advances In Data Mining
### Assignment 2
### Luit Verschuur 1811053, Marcel Kolenbrander 1653415.


import numpy as np
from itertools import compress


class SimilarityBase:
    user_movie_matrix = ...
    user_movie_matrix_shape = ...
    similarity_output_function = ...
    random_seed = ...
    similarity_limit = ...
    signature_size = ...
    user_signatures = ...
    block_amount = ...
    block_column_size = ...
    buckets = {}

    def hash_blocks(self, user_range):
        """
        For all user signatures, generated either through min hashing or random projections, hash them into buckets.
        Hashing to buckets is done by splitting the user signatures into smaller blocks of a certain row size.
        These blocks are then hashed and this hash is used to indicate a position in the bucket table. On the position
        in the hash table, the user id (or row number in this case) is appended.

        :param user_range: Indicates which range of users to make block hashes for.
        :return:
        """

        # Loop through all user rows (or rather their signatures) in a given range.
        for r in user_range:
            # For each signature, divide them into blocks of a certain block column size.
            # Hash each block and add to the bucket table (a dictionary essentially).
            for block in range(0, self.block_amount):
                # Calculate the hash for the given block.
                block_hash = hash(tuple(self.user_signatures[r][block*self.block_column_size:block*self.block_column_size+self.block_column_size]))

                # If this hash occurred before, append the user to the existing bucket,
                # else add the user to a new bucket.
                if (bucket := self.buckets.get(block_hash, None)) is not None:
                    bucket.append(r)
                else:
                    self.buckets[block_hash] = [r]

    def reduce_buckets(self):
        """
        Reduce the bucket table by removing singular elements and convert to a list.

        :return:
        """

        self.buckets = [list(set(el)) for el in list(filter(lambda dict_value: len(dict_value) > 1, self.buckets.values()))]

    def find_similar_users(self, similarity_function, bucket_range=None):
        """
        Users who share the same hash bucket in the bucket table are similarity candidates.
        For each candidate pair in the buckets, check their actual similarity using the similarity function
        provided as a parameter. If the similarity reaches above a certain limit, output them to a file.

        :param similarity_function: The similarity function to check to users against.
        :return:
        """

        print(bucket_range)

        # For each bucket loop through their items (note that the items are lists of user ids).
        for bucket_i in bucket_range if bucket_range is not None else range(0, len(self.buckets)):
            bucket = self.buckets[bucket_i]
            bucket_len = len(bucket)

            # If a bucket is of length 1 (or smaller) then only one user exists in this bucket.
            # This user does not have any comparison pairs and can therefore be ignored.
            if bucket_len > 1:

                # Loop through all combination pairs: [k(k-1)/2]
                # and check whether they exceed the similarity threshold.
                for i in range(0, bucket_len):
                    for j in range(i + 1, bucket_len):
                        if (sim := similarity_function(bucket[i], bucket[j])) > self.similarity_limit:
                            self.similarity_output_function(bucket[i], bucket[j])
                            print(f"Pair {bucket[i]}:{bucket[j]} are similar ({sim})")

    def __init__(self, user_movie_matrix_in=None, similarity_output_function_in=None, random_seed_in=None,
                 similarity_limit_in=None, signature_size_in=None, block_amount_in=None, block_row_size_in=None, *kwargs):
        if user_movie_matrix_in is not None:
            self.user_movie_matrix = user_movie_matrix_in
        else:
            raise ValueError("User Movie Matrix not initiated")

        self.user_movie_matrix_shape = self.user_movie_matrix.get_shape()

        if similarity_output_function_in is not None:
            self.similarity_output_function = similarity_output_function_in
        else:
            raise ValueError("File output function not initiated")

        if random_seed_in is not None:
            self.random_seed = random_seed_in
        else:
            raise ValueError("Random seed not initiated")

        if similarity_limit_in is not None:
            self.similarity_limit = similarity_limit_in
        else:
            raise ValueError("Similarity limit not initiated")

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

        if self.block_amount * self.block_column_size > self.signature_size:
            raise ValueError("Hash blocks exceed the signature size!")

        # Initialize the signatures as an empty 2D matrix of users and a corresponding signature vector for each user.
        self.user_signatures = np.empty(shape=(self.user_movie_matrix_shape[0], self.signature_size))
        self.user_signatures[:] = np.NaN

        # Set random seed
        np.random.seed(self.random_seed)