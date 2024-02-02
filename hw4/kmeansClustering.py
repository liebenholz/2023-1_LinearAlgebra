import sys
import numpy as np

#   Code for k-means clustering 5-vectors in the input file into k groups.
#   e.g. for the case of ten 5-vectors of [(0, 0, 0, 0, i) for i in range(1, 11)] and k=2

#   input : 
#   $python kmeansClustering.py inputfile.txt 2 10

#   output :
#   # of actual iteration : 2
#   representative : 
#   (0, 0, 0, 0, 2)
#   (0, 0, 0, 0, 7)
#   # of vectors for cluster 1 : 5
#   # of vectors for cluster 2 : 5

# Read 5-vectors from file.
def read_vectors(file_path):
    vectors = []
    with open(file_path) as f:
        for line in f:
            # Split elements by space.
            vector = line.strip().split()
            # Save each elements as float in vector array.
            vector = [float(x) for x in vector]
            # Append vector list to vectors array.
            vectors.append(vector)
    # Return vectors as numpy array.
    return np.array(vectors)

# Initiate : Select random cluster centroid.
def init_centroids(data, k):
    # Shuffle.
    np.random.shuffle(data)
    # Return k datas as first centroids.
    return data[:k,:]

# Calculate distance of each data point and centroid.
def compute_distances(data, centroids):
    # Initiate distances.
    distances = np.zeros((data.shape[0], centroids.shape[0]))
    for i in range(centroids.shape[0]):
        # Update Distances of each data and centroid by norm. 
        distances[:,i] = np.linalg.norm(data - centroids[i,:], axis=1)
    return distances

# Find closest cluster.
def find_closest_cluster(distances):
    # Data with minimum distance with centroid
    # may be included to its cluster.
    return np.argmin(distances, axis=1)

# Compute new centroids.
def compute_centroids(data, clusters, k):
    # Initiate centroids.
    centroids = np.zeros((k, data.shape[1]))
    # For k-times of iteration
    for i in range(k):
        # Find new centroids and choose groups of each data.
        centroids[i,:] = np.mean(data[clusters == i,:], axis=0)
    # return computed centroids.
    return centroids

# k-means qlgorithm.
def kmeans(data, k, max_iters):
    # Initiate : Select random cluster centroid.
    centroids = init_centroids(data, k)
    for i in range(max_iters):
        # Calculate distance of each data point and centroid.
        distances = compute_distances(data, centroids)
        # Find closest cluster.
        clusters = find_closest_cluster(distances)
        # Find new centroids.
        new_centroids = compute_centroids(data, clusters, k)
        # Returns True and break the loop 
        # if new centroids and existing centroids are element-wise equal.
        if np.allclose(new_centroids, centroids):
            print("# of actual iteration : ", i)
            break
        # if not, update centroids as new.
        centroids = new_centroids
    return centroids, clusters

# Print results 
# with current centroid vectors and clusters
def print_results(centroids, clusters):
    print("representative :")
    # The representatives of each cluster
    for centroid in centroids:
        print(tuple(centroid))
    # The number of vectors in each cluster
    for i in range(len(centroids)):
        print("# of vectors for cluster", i+1, ":", np.sum(clusters == i))

# Input format in terminal
# $python kmeansClustering.py inputfile.txt [value of k][# of iteration]
if __name__ == '__main__':
    # The first input is a file (named "inputfile.txt") that contains 5-vectors
    file_path = sys.argv[1]
    # The second input is the value of k (the number of groups)
    k = int(sys.argv[2])
    # The last input is the maximum number of iterations
    max_iters = int(sys.argv[3])
    
    # Read vectors from "inputfile.txt"
    data = read_vectors(file_path)
    # Activate k-means algorithm with input data
    centroids, clusters = kmeans(data, k, max_iters)
    # Print results
    print_results(centroids, clusters)