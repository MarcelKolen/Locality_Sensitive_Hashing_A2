class SimilarityBase:
    user_movie_matrix = ...
    similarity_output_function = ...
    random_seed = ...

    def __init__(self, user_movie_matrix_in=None, similarity_output_function_in=None, random_seed_in=None, *kwargs):
        if user_movie_matrix_in is not None:
            self.user_movie_matrix = user_movie_matrix_in
        else:
            raise ValueError("User Movie Matrix not initiated")

        if similarity_output_function_in is not None:
            self.similarity_output_function = similarity_output_function_in
        else:
            raise ValueError("File output function not initiated")

        if random_seed_in is not None:
            self.random_seed = random_seed_in
        else:
            raise ValueError("Random seed not initiated")