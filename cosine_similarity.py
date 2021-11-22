### Advances In Data Mining
### Assignment 2
### Luit Verschuur 1811053, Marcel Kolenbrander 1653415.


import numpy as np
import math
from scipy.sparse import csr_matrix

from similarity_setup import SimilarityBase


class CosineSimilarityBase(SimilarityBase):
    def __cosine_similarity(self, matrix_a, matrix_b, size_a, size_b):
        distance = np.vdot(np.asarray(matrix_a), np.asarray(matrix_b))
        size = (int(math.sqrt(size_a)) * int(math.sqrt(size_b))) * 25 # CS has values higher than 1 so 1x1 reaches to 5x5 so we have to compensate with 25

        cos_dist = math.degrees(math.acos(distance / size))

        return 1 - (cos_dist/180)

    def __generate_random_projections(self, user_range):
        column_range_max = self.user_movie_matrix.get_shape()[1]
        dense_matrix = self.user_movie_matrix.todense()
        for i in range(0, 20):
            for j in range(0, 20):
                print("i: " + str(i) + " j: " + str(j))
                print(self.__cosine_similarity(dense_matrix[i], dense_matrix[j], self.user_movie_matrix[i].size, self.user_movie_matrix[j].size))

    def __call__(self, *args, **kwargs):
        print("Now running the Cosine Similarity Routine")

        # Print found pair to file (File path is pre-set)

        # Hier moet de routine komen voor de CS methode.
        self.__generate_random_projections((0,20))

        return
