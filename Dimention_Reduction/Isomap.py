import numpy as np
import scipy
from scipy.spatial import distance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def GetEuclideanDistance(data):
    dist = distance.squareform(distance.pdist(data, 'euclidean'))
    return(dist)

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

def FloydWarshall(graph):
    n = graph.shape[0]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if graph[i,j] > graph[i,k] + graph[k,j]:
                    graph[i, j] = graph[i,k] + graph[k,j]
    return(graph)

def GetGramFromDistSquared(dist):
    n = dist.shape[0]
    P = np.eye(n) - 1/n * np.matmul( np.ones((n,1)), np.ones((1,n)) )
    G = np.matmul ( P, dist )
    G = -1/2 * np.matmul( G, P )
    return(G)

def MDS(G, m):
    w, Q = np.linalg.eig(G)
    Q = Q[:,scipy.argsort(w)[::-1][:m]]
    D = np.diag(w[scipy.argsort(w)[::-1][:m]])
    Dstar = np.sqrt(D)
    y = np.matmul(Q, Dstar)
    return(y)

def Isomap(data, m):
    K = 10
    n, d = data.shape
    dist = GetEuclideanDistance(data)
    graph = GetKnnGraph(dist, K)
    shortDist = FloydWarshall(graph)
    shortDist2 = shortDist * shortDist  ## need to use squared distance to construct gram!!!!!!
    gram = GetGramFromDistSquared(shortDist2)
    mappedData = MDS(gram, m)
    return(mappedData)



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

    # Isomap
    reduced = Isomap(data[:, 0:3], 2)
    plt.scatter(reduced[data[:, 3] == 1, 0], reduced[data[:, 3] == 1, 1], color="green")
    plt.scatter(reduced[data[:, 3] == 2, 0], reduced[data[:, 3] == 2, 1], color="yellow")
    plt.scatter(reduced[data[:, 3] == 3, 0], reduced[data[:, 3] == 3, 1], color="blue")
    plt.scatter(reduced[data[:, 3] == 4, 0], reduced[data[:, 3] == 4, 1], color="red")
    plt.title("Isomap results")
    plt.show()



