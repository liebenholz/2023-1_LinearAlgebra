import sys
import numpy as np

# Function whether the set of input 20-vectors are linearly independent
# by using Gram-Schmidt Algorithm.
def is_linearly_independent(vectors):
    q_vectors = np.zeros_like(vectors)  # orthogonal vectors.
    n_checked = 0  # number of vectors that have been checked.

    for i in range(vectors.shape[0]):
        # Compute orthogonal vector q_i with
        # (i-1)th orthogonal vector and input; vectors[i].
        if i == 0:
            # Initialization
            q_vectors[i] = vectors[i] / np.linalg.norm(vectors[i])
        else:
            for j in range(i):
                q_vectors[i] += np.dot(vectors[i], q_vectors[j]) / np.linalg.norm(q_vectors[j])**2 * q_vectors[j]
            # q_i = a_i - q_(i-1)
            q_vectors[i] = vectors[i] - q_vectors[i]
            # If two arrays are element-wise equal with 0 vector
            if np.allclose(q_vectors[i], np.zeros_like(q_vectors[i])):
                # Return False(linearly dependent) and number of vectors that have been checked.
                return False, n_checked + 1
            else: # Continue, Normalization
                q_vectors[i] = q_vectors[i] / np.linalg.norm(q_vectors[i])
        # Update number of vectors that have been checked.
        n_checked += 1
    # Return True(linearly independent) and number of vectors that have been checked.
    return True, n_checked


# $python kmeansClustering.py inputfile.txt [value of k][# of iteration]
if __name__ == '__main__':
    # The first input is a file (named "inputfile.txt") that contains 5-vectors
    file_path = sys.argv[1]
    # Read 20-vectors form "inputfile.txt" and make them into numpy array.
    vectors = np.loadtxt(file_path)

    # The result that indicates whether the set of input vectors is 
    # linearly independent or linearly dependent.
    is_independent, n_checked = is_linearly_independent(vectors)
    if is_independent:
        print("Linearly independent.")
    else:
        print("Linearly dependent.")
    # The number of vectors that have checked until the decision is made.
    print("Number of vectors checked: ", n_checked)
