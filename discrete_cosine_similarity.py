### Advances In Data Mining
### Assignment 2
### Luit Verschuur 1811053, Marcel Kolenbrander 1653415.


import numpy as np
import math
import time

from scipy.sparse import csr_matrix
from numpy import linalg as LA

from parallels import Parallels
from similarity_setup import SimilarityBase
from cosine_similarity import CosineSimilarityBase


class DiscreteCosineSimilarityBase(CosineSimilarityBase):
    binary_matrix = []
    sqrt_non_zero_count_binary_matrix = []

    def calculate_user_similarity(self, usr_0, usr_1):
        """

        :param usr_0: A users ID
        :param usr_1: A users ID
        :return: The discrete cosine similarity between 0. and 1. (inclusive boundaries)
        """

        if usr_0 == usr_1:
            return 0

        distance = self.binary_matrix.getrow(usr_0).dot(self.binary_matrix.getrow(usr_1).transpose()).toarray()[0][0]
        size = self.sqrt_non_zero_count_binary_matrix[usr_0] * self.sqrt_non_zero_count_binary_matrix[usr_1]

        return 1 - (math.degrees(math.acos(distance / size))/180)

    def init(self, *args, **kwargs):
        # Convert matrix into a binary matrix where all 0 values remain zero and all non zero values are 1.
        self.binary_matrix = self.user_movie_matrix.copy()
        self.binary_matrix.data = np.ones_like(self.binary_matrix.data)

        # Get the square root of number of non zero elements per row.
        self.sqrt_non_zero_count_binary_matrix = np.sqrt(self.binary_matrix.getnnz(axis=1))

        self.welcome_text = "Now running the Discrete Cosine Similarity Routine"
