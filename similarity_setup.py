### Advances In Data Mining
### Assignment 2
### Luit Verschuur 1811053, Marcel Kolenbrander 1653415.


import numpy as np
import time

from parallels import Parallels

class SimilarityBase:
    user_movie_matrix = ...
    user_movie_matrix_shape = ...
    similarity_output_function = ...
    result_file_name = ...
    random_seed = ...
    similarity_limit = ...
    signature_size = ...
    user_signatures = ...
    block_amount = ...
    block_column_size = ...
    buckets = {}
    welcome_text = "Now running the GENERIC Similarity Routine"
    number_of_processes = ...

    def output_pair_to_file(self, usr_0, usr_1):
        self.similarity_output_function(usr_0, usr_1, out_file_name=self.result_file_name)

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
        Reduce individual elements by removing duplicate entries.

        :return:
        """

        self.buckets = [list(set(el)) for el in list(filter(lambda dict_value: len(dict_value) > 1, self.buckets.values()))]

    @staticmethod
    def __sort_func(e):
        return len(e)

    def sort_buckets(self, reversed=False):
        """
        Sort the buckets by number of elements in each bucket from smallest to biggest
        (or biggest to smallest if reversed is true).
        If the elements in the following example list the number of keys stored in each bucket:
        [2, 3, 2, 5, 3, 7, 9, 4]
        Then the sorted variant is:
        [2, 2, 3, 3, 4, 5, 7, 9]
        If reversed is True, then the sorted variant is reversed (biggest to smallest).

        :param reversed: Indicate whether the buckets are
        :return:
        """

        self.buckets = sorted(self.buckets, key=self.__sort_func, reverse=reversed)

    def re_arrange_buckets(self, chunks=1):
        """
        Re arrange the buckets into alternating chunks.
        If the following example list is the input list:
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        Then a re arranged list with chunk size 4 is:
        [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]

        :param chunks: Number of chunks to divide the dataset in
        :return: Chunk boundary areas
        """

        re_arranged_buckets = [self.buckets[i::chunks] for i in range(0, chunks)]

        chunk_borders = [0] * len(re_arranged_buckets)

        last_chunk_border = 0

        for i, chunk in enumerate(re_arranged_buckets):
            current_chunk_border = len(chunk)

            chunk_borders[i] = range(last_chunk_border, current_chunk_border + last_chunk_border)

            last_chunk_border += current_chunk_border

        # Unpack chunks back into a list.
        self.buckets = [item for sublist in re_arranged_buckets for item in sublist]

        return chunk_borders

    def find_similar_users(self, bucket_range, similarity_function):
        """
        Users who share the same hash bucket in the bucket table are similarity candidates.
        For each candidate pair in the buckets, check their actual similarity using the similarity function
        provided as a parameter. If the similarity reaches above a certain limit, output them to a file.

        :param similarity_function: The similarity function to check to users against.
        :return:
        """

        # For each bucket loop through their items (note that the items are lists of user ids).
        for bucket_i in bucket_range:
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
                            self.output_pair_to_file(bucket[i], bucket[j])
                            print(f"Pair {bucket[i]}:{bucket[j]} are similar ({sim})")

    def bucket_metrics(self):
        """
        Calculate metrics such as:
        - Average size of buckets
        - Min size of buckets
        - Max size of buckets
        :return:
        """

        average_length = 0.
        min = np.inf
        max = 0.

        # Loop through all buckets to calculate metrics.
        for bucket in self.buckets:
            len_bucket = len(bucket)

            if min > len_bucket:
                min = len_bucket
            if max < len_bucket:
                max = len_bucket

            average_length += len_bucket

        average_length /= len(self.buckets)

        print(f"There are {len(self.buckets)} buckets in total. On average they have a size of {average_length} (min {min}, max {max}) \n")

    def generate_signatures_for_users(self, user_range):
        pass

    def calculate_user_similarity(self, usr_0, usr_1):
        pass

    def init(self, *args, **kwargs):
        pass

    def __init__(self, user_movie_matrix_in=None, similarity_output_function_in=None, result_file_name_in=None, random_seed_in=None,
                 similarity_limit_in=None, signature_size_in=None, block_amount_in=None, block_row_size_in=None, number_of_processes_in=None, *args, **kwargs):
        if user_movie_matrix_in is not None:
            self.user_movie_matrix = user_movie_matrix_in
        else:
            raise ValueError("User Movie Matrix not initiated")

        self.user_movie_matrix_shape = self.user_movie_matrix.get_shape()

        if similarity_output_function_in is not None:
            self.similarity_output_function = similarity_output_function_in
        else:
            raise ValueError("File output function not initiated")

        if result_file_name_in is not None:
            self.result_file_name = result_file_name_in
        else:
            raise ValueError("Output file not initiated")

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

        if number_of_processes_in is not None:
            self.number_of_processes = number_of_processes_in
        else:
            self.number_of_processes = 1

        # Set random seed.
        np.random.seed(self.random_seed)

        # Call init for the inherit classes.
        self.init(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        # Display the current running routine.
        print(self.welcome_text)

        # Define size of problem space.
        total_user_space = int(self.user_movie_matrix_shape[0])

        print("Generating signatures")
        start_time = time.perf_counter()

        # Split between running in parallel or on a single process.
        if self.number_of_processes > 1:
            # Initialize parallel workers.
            pa = Parallels(target_function_in=self.generate_signatures_for_users,
                           max_number_of_workers_in=self.number_of_processes)

            # Find user signatures (min hashing/random projections) and store results in user_signatures.
            # This process is split up and divided over workers by mapping chunks of the problem space. Each worker
            # returns a subset of the solution, which have to be concatenated into one solution.
            self.user_signatures = np.concatenate(tuple(i[1] for i in sorted(pa.run_map(
                [range(int(i * total_user_space / self.number_of_processes), int((i + 1) * total_user_space / self.number_of_processes)) for i in
                 range(0, self.number_of_processes)]).items(), key=lambda i: i[0])), axis=0)
        else:
            # Find user signatures (min hashing/random projections) and store results in user_signatures.
            self.user_signatures = self.generate_signatures_for_users(range(0, total_user_space))
        print(f"Generating signatures took: {time.perf_counter() - start_time}\n")


        print("Generating hashes")
        start_time = time.perf_counter()
        # Hash the user signatures into buckets.
        self.hash_blocks(range(0, total_user_space))
        print(f"Generating hashes took: {time.perf_counter() - start_time}\n")

        # Remove buckets with only one user in it.
        self.reduce_buckets()

        # Sort buckets from smallest to largest.
        self.sort_buckets()

        # Information on buckets.
        self.bucket_metrics()

        print("Evaluating similarity canidates")
        start_time = time.perf_counter()

        if self.number_of_processes > 1:
            # Re arrange the buckets based on number of processes to optimize the load between the different workers.
            # Each worker gets a slice of the problem space (buckets) with a roughly equal distribution from small to large
            # buckets between all workers.
            # If the following example list is the input list where each element represents the length of a bucket:
            # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            # Then a re arranged list with chunk size 4 (number of processes) is:
            # [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
            chunk_borders = self.re_arrange_buckets(self.number_of_processes)

            pa = Parallels(target_function_in=self.find_similar_users, max_number_of_workers_in=self.number_of_processes)
            pa.run_map(chunk_borders, (self.calculate_user_similarity, ))
        else:
            self.find_similar_users(range(0, len(self.buckets)), self.calculate_user_similarity)
        print(f"Evaluating similarity canidates took: {time.perf_counter() - start_time}\n")
