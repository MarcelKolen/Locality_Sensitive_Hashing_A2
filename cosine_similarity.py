### Advances In Data Mining
### Assignment 2
### Luit Verschuur 1811053, Marcel Kolenbrander 1653415.


import numpy as np
import math
import time

from scipy.sparse import csr_matrix
from scipy.sparse import linalg as spla
from numpy import linalg as LA

from parallels import Parallels
from similarity_setup import SimilarityBase


class CosineSimilarityBase(SimilarityBase):
    linalg_norm = []

    def calculate_user_similarity(self, usr_0, usr_1):
        """

        :param usr_0: A users ID
        :param usr_1: A users ID
        :return: The cosine similarity between 0. and 1. (inclusive boundaries)
        """
        if usr_0 == usr_1:
            return 0

        distance = self.user_movie_matrix.getrow(usr_0).dot(self.user_movie_matrix.getrow(usr_1).transpose()).toarray()[0][0]
        size = self.linalg_norm[usr_0] * self.linalg_norm[usr_1]

        cos_dist = math.degrees(math.acos(distance / size))
        return 1 - (cos_dist/180)

    def generate_signatures_for_users(self, user_range):
        nbits = self.user_movie_matrix_shape[1]

        user_signatures = np.empty(shape=(user_range.stop - user_range.start, self.signature_size))
        user_signatures[:] = np.NaN

        plane_norms = np.random.rand(nbits, self.signature_size) - 0.5
        for r in user_range:
            rand_proj = self.user_movie_matrix.getrow(r).dot(plane_norms)[0] > 0
            for i in range(0, self.signature_size):
                user_signatures[r - user_range.stop, i] = rand_proj[i]

        return user_signatures

    def init(self, *args, **kwargs):
        # Get the linear algebra norm for each row.
        self.linalg_norm = spla.norm(self.user_movie_matrix, axis=1)

        self.welcome_text = "Now running the Cosine Similarity Routine"
