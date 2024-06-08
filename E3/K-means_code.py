from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt


# Load dataset
data = load_iris()
X = data.data

# Initialization 
K = 3
runs = 5



# K-means algorithm

def kmeans(centroids, X, max_iter=1000):
    SSE = []
    for _ in range(max_iter):
        # Assign each data point to the closest centroid
        clusters = np.argmin(np.linalg.norm(X[:, None] - centroids, axis=2), axis=1)

        # Update the centroids
        new_centroids = np.array([X[clusters == k].mean(axis=0) for k in range(K)])

        # Check if the centroids have converged
        if np.all(centroids - new_centroids < 1e-8):
                break

        centroids = new_centroids
        SSE.append(SSE_func(centroids, X))
    
    

    return centroids, clusters, SSE
        
def SSE_func(centroids, X):
    sse = np.sum(np.min(np.linalg.norm(X[:, None] - centroids, axis=2), axis=1))
    return sse





def main():
    for i in range(runs):
        # Randomly initialize the centroids
        centroids = X[np.random.choice(X.shape[0], K, replace=False)]

        # Run the K-means algorithm
        centroids, cluster, SSE = kmeans(centroids, X)
        print(f'Run {i+1}')
        print(f'Final Centroids:\n {centroids}')
        print(f'best_labels:\n {cluster}')
        print(f'SSE: {SSE[-1]}')
        

        plt.figure()
        plt.plot(range(len(SSE)), SSE)
        plt.xlabel('Iteration')
        plt.ylabel('SSE')
        plt.title(f'Run:{i+1}\nSSE vs Iteration')
    
    plt.show()




if __name__ == '__main__':
    main()