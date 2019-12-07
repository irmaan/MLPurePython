import numpy as np
import scipy
from scipy.spatial import distance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# get euclidean distance between each pair of data points
def GetEuclideanDistance(data):
    dist = distance.squareform(distance.pdist(data, 'euclidean'))
    #dist = dist * dist
    return(dist)

# get k nearest neighbors graph
def GetKnnGraph(dist, K):
    n = dist.shape[0]
    graph = -1 * np.ones(dist.shape)
    nearest = list(scipy.argsort(dist, axis=1)[:,:K])
    rows = list(np.array(range(n)).reshape((n,1)) * np.ones((n, K), dtype='int'))
    graph[rows, nearest] = dist[rows, nearest]
    Insym = graph * np.transpose(graph)
    graph[Insym<0] = -1 * Insym[Insym<0]
    graph[graph<0] = float('Inf')
    return(graph)

# find weights and express each data point as a linear combination of its k nearest neighbors
def FindWeights(data, knnGraph):
    n,d = data.shape
    epsilon = 1e-3
    W = np.zeros((n,n))
    for i in range(n):
        neighbor = data[ np.isfinite(knnGraph[i,:]), : ]
        K = np.matmul( neighbor-data[i,:], np.transpose(neighbor-data[i,:]) )
        m = K.shape[0]
        Kinv = np.linalg.inv(K+epsilon*np.eye(m))
        w = np.matmul(Kinv, np.ones((m,1)))
        w = w / sum(w)
        W[i, np.isfinite(knnGraph[i, :])] = w.reshape((1,m))
    return(W)

# find mapped data that retain the weights
def FindData(W, p):
    n = W.shape[0]
    I = np.eye(n)
    M = np.matmul( I-np.transpose(W), I-W)
    w, v = np.linalg.eig(M)
    w = list(np.argsort(w))[1:(p+1)]
    Y = v[:,w] * np.sqrt(n)
    return(Y)

def LLE(data, m):
    K = 10
    n, d = data.shape
    dist = GetEuclideanDistance(data)
    graph = GetKnnGraph(dist, K)
    W = FindWeights(data, graph)
    Y = FindData(W, m)
    return(Y)


if __name__ == '__main__':
    data = np.loadtxt("./3Ddata.txt", )
    n,d = data.shape

    # plot data in 3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(xs=data[data[:,3]==1,0],ys=data[data[:,3]==1 ,1],zs=data[data[:,3]==1 ,2], c="green")
    ax.scatter(xs=data[data[:, 3] == 2, 0], ys=data[data[:, 3] == 2, 1], zs=data[data[:, 3] == 2, 2], c="yellow")
    ax.scatter(xs=data[data[:, 3] == 3, 0], ys=data[data[:, 3] == 3, 1], zs=data[data[:, 3] == 3, 2], c="blue")
    ax.scatter(xs=data[data[:, 3] == 4, 0], ys=data[data[:, 3] == 4, 1], zs=data[data[:, 3] == 4, 2], c="red")
    plt.show()

    # LLE
    reduced = LLE(data[:, 0:3], 2)
    plt.scatter(reduced[data[:, 3] == 1, 0], reduced[data[:, 3] == 1, 1], color="green")
    plt.scatter(reduced[data[:, 3] == 2, 0], reduced[data[:, 3] == 2, 1], color="yellow")
    plt.scatter(reduced[data[:, 3] == 3, 0], reduced[data[:, 3] == 3, 1], color="blue")
    plt.scatter(reduced[data[:, 3] == 4, 0], reduced[data[:, 3] == 4, 1], color="red")
    plt.title("LLE results")
    plt.show()

