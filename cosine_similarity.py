import numpy as np
import math
from scipy.sparse import csr_matrix
from sklearn.random_projection import GaussianRandomProjection
from numpy import linalg as LA

from similarity_setup import SimilarityBase


class CosineSimilarityBase(SimilarityBase):
    random_projections = []
    nbits = 17770 # Must be equal to the number of columns

    def __cosine_similarity(self, matrix_a, matrix_b, size_a, size_b):

        distance = np.vdot(np.asarray(matrix_a), np.asarray(matrix_b))
        size = size_a * size_b
        cos_dist = math.degrees(math.acos(distance / size))

        return 1 - (cos_dist/180)

    def __generate_random_projections(self, user_range):
        dim = 4 # Variable to play with
        column_range_max = self.user_movie_matrix.get_shape()[1]
        dense_matrix = self.user_movie_matrix.todense()
        random_projections = []

        plane_norms = np.random.rand(self.nbits, dim) - 0.5
        print(plane_norms)
        for i in range(0, column_range_max):
            self.random_projections.append(np.dot(np.asarray(dense_matrix[i]), plane_norms))

        self.random_projections = np.asarray(self.random_projections) > 0
        # print(self.random_projections[0])


        # NU LIMITED TOT GECOMPRIMEERDE 20, OPLETTEN BIJ IMPLEMENTATIE
        # PS: OOK EEN OPTIE OM EERST ALLE SIZES TE ACHTERHALEN, DAN DOEN WE NOG MINDER REDUNDANT SHIT. VOOR NU LAAT IK DIT FF ZO
        #for i in range(0, 20):
        #    size_i = LA.norm(dense_matrix[i])
        #    for j in range(i+1, 20):
        #        if i != j:
        #            size_j = LA.norm(dense_matrix[j])
        #            print("i: " + str(i) + " j: " + str(j))
        #            print(self.__cosine_similarity(dense_matrix[i], dense_matrix[j], size_i, size_j))



    def __call__(self, *args, **kwargs):


        print("Now running the Cosine Similarity Routine")

        # Print found pair to file (File path is pre-set)
        # DIT IS IN PRINCIPE DE RANDOM PROJECTION, MAAR WAT DOEN WE MET DIE VECTOR?!?!?!?!



        #X = rng.rand(100, 10000) #SIZE?!?!?!!?!
        #transformer = GaussianRandomProjection(random_state=rng)
        #X_new = transformer.fit_transform(self.user_movie_matrix)
        #print(X_new.shape)



        # Hier moet de routine komen voor de CS methode.
        self.__generate_random_projections((0,20))



        return
