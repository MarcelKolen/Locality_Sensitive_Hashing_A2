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
        Calculate the cosine similarity (a value between 0. and 1.) of two given users. Compare the users using
        their original data entries. The cosine similarity is the angle between two users represented as
        vectors.

        :param usr_0: A users ID
        :param usr_1: A users ID
        :return: The cosine similarity between 0. and 1. (inclusive boundaries)
        """

        if usr_0 == usr_1:
            return 0

        # The distance is the dot product of the two vectors representing the users.
        distance = self.user_movie_matrix.getrow(usr_0).dot(self.user_movie_matrix.getrow(usr_1).transpose()).toarray()[0][0]

        # The size is the product of the two linear norms of the user vectors.
        size = self.linalg_norm[usr_0] * self.linalg_norm[usr_1]

        # The similarity is the difference between one and the degrees in 180 proportion of the arc cosine on the
        # distance over size.
        return 1 - (math.degrees(math.acos(distance / size))/180)

    def generate_signatures_for_users(self, user_range):
        """
        For each user (row) in a given range, calculate the signature using random projections on the columns.
        The collection of random projection border results (e.g. above the project is one, below is zero)
        form the signature for the users.

        :param user_range: A range of users (rows) over which the loop should run
        :return: An 2D array containing signatures for the users based on random projections
        """

        # Init empty user signatures of size given by the user_range and the signature size.
        user_signatures = np.empty(shape=(user_range.stop - user_range.start, self.signature_size))
        user_signatures[:] = np.NaN

        # The number of values to create the plane norms over (the number of movies per user)
        nbits = self.user_movie_matrix_shape[1]

        # The plane norms contain random values between -0.5 and 0.5 ([0, 1) - 0.5) of size #movies * signature_size.
        plane_norms = np.random.rand(nbits, self.signature_size) - 0.5

        # For each users (row) calculate the signature using random projection.
        for r in user_range:
            # The random projections, rather user signatures, are the dot product of a user vector and the plane norms.
            # Note that the result is converted to binary (True(1)/False(0)) values. Every positive non zero value is
            # set to 1 and all others are 0.
            user_signatures[r - user_range.stop] = self.user_movie_matrix.getrow(r).dot(plane_norms)[0] > 0

        return user_signatures

    def init(self, *args, **kwargs):
        # Get the linear algebra norm for each row.
        self.linalg_norm = spla.norm(self.user_movie_matrix, axis=1)

        self.welcome_text = "Now running the Cosine Similarity Routine"
