import numpy as np


class SimilarityBase:
    user_movie_matrix = ...
    user_movie_matrix_shape = ...
    similarity_output_function = ...
    random_seed = ...
    similarity_limit = ...

    def __init__(self, user_movie_matrix_in=None, similarity_output_function_in=None, random_seed_in=None, similarity_limit_in=None, *kwargs):
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

        # Set random seed
        np.random.seed(self.random_seed)