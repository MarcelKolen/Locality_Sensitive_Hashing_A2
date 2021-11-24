### Advances In Data Mining
### Assignment 2
### Luit Verschuur 1811053, Marcel Kolenbrander 1653415.


import numpy as np
import math
import time

from scipy.sparse import csr_matrix
from numpy import linalg as LA

from similarity_setup import SimilarityBase


class CosineSimilarityBase(SimilarityBase):
    dense_matrix = []
    linalg_norm = []

    def __calculate_cosine_similarity(self, usr_0, usr_1):
        if usr_0 == usr_1:
            return 0

        distance = np.vdot(np.array(self.user_movie_matrix.getrow(usr_0).toarray()), np.array(self.user_movie_matrix.getrow(usr_1).toarray()))
        size = self.linalg_norm[usr_0] * self.linalg_norm[usr_1]

        cos_dist = math.degrees(math.acos(distance / size))
        return 1 - (cos_dist/180)

    def __generate_random_projections(self, user_range):
        nbits = self.user_movie_matrix_shape[1]

        plane_norms = np.random.rand(nbits, self.signature_size) - 0.5
        for r in user_range:
            rand_proj = np.asarray(np.dot(np.asarray(self.dense_matrix[r]), plane_norms)) > 0
            for i in range(0, self.signature_size):
                self.user_signatures[r, i] = rand_proj[0][i]


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dense_matrix = self.user_movie_matrix.todense()
        for i in range(0, self.dense_matrix.shape[0]):
            self.linalg_norm.append(LA.norm(self.dense_matrix[i]))


    def __call__(self, *args, **kwargs):
        print("Now running the Cosine Similarity Routine")

        print("Generating random projections")
        start_time = time.perf_counter()
        self.__generate_random_projections(range(0, int(self.user_movie_matrix_shape[0])))
        print(f"Generating random projections took: {time.perf_counter() - start_time}\n")

        print("Generating hashes")
        start_time = time.perf_counter()
        self.hash_blocks(range(0, int(self.user_movie_matrix_shape[0])))
        print(f"Generating hashes took: {time.perf_counter() - start_time}\n")

        self.reduce_buckets()

        print("Evaluating similarity canidates")
        start_time = time.perf_counter()
        self.find_similar_users(self.__calculate_cosine_similarity)
        print(f"Evaluating similarity canidates took: {time.perf_counter() - start_time}\n")


        return


