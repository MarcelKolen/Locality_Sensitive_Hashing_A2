### Advances In Data Mining
### Assignment 2
### Luit Verschuur 1811053, Marcel Kolenbrander 1653415.

import sys
import numpy as np
from scipy.sparse import csr_matrix
from enum import Enum, auto

from jaccard_similarity import JaccardSimilarityBase
from cosine_similarity import CosineSimilarityBase
from discrete_cosine_similarity import DiscreteCosineSimilarityBase


class SimilarityMeasureOptions(Enum):
    JACCARD = auto()
    COSINE = auto()
    DISCRETE_COSINE = auto()


similarity_measure = None
data_path = None
random_seed = None


def check_sys_arg_bounds(pos):
    # Check if pos +1 is not out of bounds
    # (Needed to get parameter values with parameter flags)
    if (pos := pos + 1) < len(sys.argv):
        return pos

    print("ERROR: parameters not correct. Use -h or -? to show help...")
    exit(-1)


def argument_data_path(pos):
    global data_path

    # Fetch data path from command line input
    data_path = str(sys.argv[check_sys_arg_bounds(pos)])


def argument_seed(pos):
    global random_seed

    # Fetch random seed from command line input
    random_seed = int(sys.argv[check_sys_arg_bounds(pos)])


def argument_help(*kwargs):
    print(f"Use: python main.py [parameters]\n"
          f"Parameter options:\n"
          f" * [-h str] -> show this help screen\n"
          f" * [-d str] -> set data path (str is path)\n"
          f" * [-s int] -> set random seed for numpy.random.seed(int) (int is seed)\n"
          f" * [-m str] -> select the similarity measure mode. Options:\n"
          f"   * 'js' Jaccard similarity with minhashing\n"
          f"   * 'cs' Cosine similarity with random projections\n"
          f"   * 'dcs' Discrete Cosine similarity with random projections\n")
    exit(0)


mode_options = {
    'js': SimilarityMeasureOptions.JACCARD,
    'cs': SimilarityMeasureOptions.COSINE,
    'dcs': SimilarityMeasureOptions.DISCRETE_COSINE
}


def argument_mode(pos):
    global similarity_measure

    # Check whether the provided similarity measure is known with the programme
    if (similarity_measure := mode_options.get(sys.argv[check_sys_arg_bounds(pos)], None)) is None:
        print("ERROR: Similarity measure unknown. Use -h or -? to show help...")
        exit(-1)


argument_options = {
    '-d': argument_data_path,
    '-s': argument_seed,
    '-m': argument_mode,
    '-h': argument_help,
    '-?': argument_help
}


def process_arguments():
    if len(sys.argv) <= 1:
        argument_help()

    # For each program argument, handle if flag is found.
    for i, arg in enumerate(sys.argv):
        # Sort of switch case (please Python 3.10 hurry up, we need switches).
        if (call := argument_options.get(arg)) is not None:
            call(i)


def write_pair_to_file(uid_0, uid_1, out_file_name="default.txt"):
    # Write user id pair to an output file by appending the file.
    with open(out_file_name, "a") as out_file:
        # Check whether uid_0 < uid_1, if not: swap.
        out_file.write(f"{f'{uid_0}, {uid_1}' if uid_0 < uid_1 else f'{uid_1}, {uid_0}'}\n")
        out_file.close()

    return


def main():
    # Process arguments
    process_arguments()

    if data_path is None:
        print("Data file not initialized")
        exit(-1)

    try:
        # Load .npy file and convert into numpy array.
        data_file = np.load(data_path)

        # Convert the datafile ((user*movies) x 3) [uid, mid, rt] into a sparse CSR matrix
        # With USER_IDS as rows (note uids are decremented: original_uid=1 -> uid=0, original_uid=2 -> uid=1, etc.)
        # and MOVIE_IDS as columns (note mids are decremented: original_mid=1 -> mid=0, original_mid=2 -> mid=1, etc.)
        # the elements in the sparse matrix represent the ratings.
        matrix_shape = data_file.max(axis=0)

        user_movie_sp_matrix = csr_matrix((data_file[:, 2], (data_file[:, 0] - 1, data_file[:, 1] - 1)), shape=(matrix_shape[0], matrix_shape[1]))

        # Explicitly mark for memory cleanup
        del data_file

        # Call main similarity measure classes and execute their instances to
        # find similar pairs and print to designated file.
        try:
            if similarity_measure is SimilarityMeasureOptions.JACCARD:
                js = JaccardSimilarityBase(user_movie_matrix_in=user_movie_sp_matrix, random_seed_in=random_seed,
                                           signature_size_in=150, block_amount_in=25, block_row_size_in=6,
                                           similarity_limit_in=0.5, similarity_output_function_in=write_pair_to_file,
                                           result_file_name_in="js.txt", number_of_processes_in=8)
                js()
            elif similarity_measure is SimilarityMeasureOptions.COSINE:
                cs = CosineSimilarityBase(user_movie_matrix_in=user_movie_sp_matrix, random_seed_in=random_seed,
                                          signature_size_in=150, block_amount_in=10, block_row_size_in=15,
                                          similarity_limit_in=0.73, similarity_output_function_in=write_pair_to_file,
                                          result_file_name_in="cs.txt", number_of_processes_in=8)
                cs()
            elif similarity_measure is SimilarityMeasureOptions.DISCRETE_COSINE:
                dcs = DiscreteCosineSimilarityBase(user_movie_matrix_in=user_movie_sp_matrix, random_seed_in=random_seed,
                                                   signature_size_in=225, block_amount_in=15, block_row_size_in=15,
                                                   similarity_limit_in=0.73, similarity_output_function_in=write_pair_to_file,
                                                   result_file_name_in="dcs.txt", number_of_processes_in=8)
                dcs()
            else:
                print("ERROR: Similarity measure is unkown! Use -h or -? to show help...")
                exit(-1)
        except ValueError as err:
            print(f"ERROR: Class setup failed '{err}'")

    except IOError:
        print(f"ERROR: Data file [{data_path if data_path is not None else 'FILE IS NONE'}] (.npy file) could not be read!")
        exit(-1)


if __name__ == "__main__":
    main()
