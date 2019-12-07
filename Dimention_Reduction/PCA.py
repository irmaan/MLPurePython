import numpy as np
import scipy
from scipy.spatial import distance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def CenterData(data):
    n, d = data.shape
    data = data - np.sum(data, axis=0) / n
    return(data)

def PCA(data, m):
    n, d = data.shape
    data = CenterData(data)
    Sigma = np.matmul( np.transpose(data), data) / n
    w, v = np.linalg.eig(Sigma)
    w = list(np.argsort(w)[::-1])[:m]
    PCs = v[:,w]
    reduced = np.matmul(data, PCs)
    return(reduced)



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

    # PCA
    reduced = PCA(data[:,0:3], 2)
    plt.scatter(reduced[data[:, 3]==1, 0], reduced[data[:,3]==1, 1], color="green")
    plt.scatter(reduced[data[:, 3] == 2, 0], reduced[data[:, 3] == 2, 1], color="yellow")
    plt.scatter(reduced[data[:, 3] == 3, 0], reduced[data[:, 3] == 3, 1], color="blue")
    plt.scatter(reduced[data[:, 3] == 4, 0], reduced[data[:, 3] == 4, 1], color="red")
    plt.title("PCA results")
    plt.show()


