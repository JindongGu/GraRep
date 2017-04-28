'''
This is a python implementation of the paper "GraRep: Learning Graph Representations with Global Structural Information".

In this application scenario, the proposed idea is applied to part(all friends of one user) of the datasets (social circles: facebook).

The datasets, which consists of friends of the user as nodes and friend-relationship between his or her friends as edges, can be treated as graph data.

The code try to learn a distributted representation for each node considering the global strutural information of the graph.

The learned representation is used to clustering. The clusters are different social circles of the user.

'''


#import used packages
import numpy as np
from time import time,strftime,localtime,gmtime

from autoencoder import test_dA
from kMeans import kMeans
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


#specify the hyperparameters
step = 10  ## the number of steps to collect structural information
dim = 100  ## the number of dimentionalitied of the disttributted representation
alpha = 0.5
num_clusters = 15

#load nodes of the graph
with open("graph_data/414.feat", "rb") as f:
    nodes_feat = f.readlines()

nodes = []
for node in nodes_feat:
    nodes.append(node.split(' ', 1)[0])
    
print("\nthe number of nodes is "+str(len(nodes)))

#load edges of the graph
with open("graph_data/414.edges", "rb") as f:
    nodes_pairs = f.readlines()

edges = []
for nodes_pair in nodes_pairs:
    edges.append(nodes_pair.split(' ')[0])  #every node with even index k is connected to the node with index k+1.
    edges.append(nodes_pair.split(' ')[1].split('\n')[0])
    
print("\nthe number of edges is "+str(len(edges)/2))

#get adjacency matrix
nodes_num = len(nodes)

adjacency_matrix = np.zeros((nodes_num, nodes_num))
for i in range(len(edges)/2):
    start_index = 0
    end_index = 0
    for start_node in nodes:
        if start_node == edges[2*i]:
            for end_node in nodes:
                if end_node == edges[2*i+1]:
                    adjacency_matrix[start_index][end_index] = 1
                    break
                end_index += 1
            break
        start_index += 1


#compute inverse degree matrix and transition matrix
np.fill_diagonal(adjacency_matrix, 0) #replace the diagnal elements with zeros
sum_vect = np.sum(adjacency_matrix, axis=1)

for i in range(len(sum_vect)):
    if sum_vect[i] != 0:
        sum_vect[i] = 1/sum_vect[i]
        
degree_matix_inv_like = np.diag(sum_vect)

transition_matrix = np.dot(degree_matix_inv_like, adjacency_matrix)

        
#generate k-step transition matrix
k_transition_matrix = np.zeros((nodes_num, nodes_num,step))
k_transition_matrix[:,:,0] = transition_matrix

for i in range(1, step):
    k_transition_matrix[:,:,i] = np.dot(k_transition_matrix[:,:,i-1],transition_matrix)

#calculate probability transition matrix and factorise the matrix
def GetProbTranMat(Ak):
    
    num_node, num_node2 = Ak.shape
    if (num_node != num_node2):
        print('M must be a square matrix!')
        
    Ak_sum = np.sum(Ak, axis=0).reshape(1,-1)
    Ak_sum = np.repeat(Ak_sum, num_node, axis=0)

    probTranMat = np.log(np.divide(Ak, Ak_sum)) - np.log(1./num_node)  
    probTranMat[probTranMat < 0] = 0;                   #set zero for negative and -inf elements
    probTranMat[np.isnan(probTranMat)] = 0;             #set zero for nan elements (the isolated nodes)
    
    return probTranMat


#generate representation matrix in case that the matrix is factorised with svd method
representation_matrix_svd = np.zeros((nodes_num, step*dim))
def GetRepMat_svd():
    for i in range(step):
        probTranMat = GetProbTranMat(k_transition_matrix[:,:,i])

        U, S, V = np.linalg.svd(np.float32(probTranMat), full_matrices=1, compute_uv=1)
        Ud = U[:, :dim]
        S = np.diag(S)
        Sd = S[:dim, :dim]
        
        Rk = np.dot(Ud, np.power(Sd, alpha))
        Rk = np.divide(Rk, np.repeat(np.sqrt(np.sum(np.power(Rk, 2), axis=1)).reshape(-1,1), dim, axis=1))
        representation_matrix_svd[:, dim*(i):dim*(i+1)] = Rk;

    #for the isolated node whose feature factor fulls Inf and NaN because of normalisation
    representation_matrix_svd[np.isnan(representation_matrix_svd)] = 0
    representation_matrix_svd[np.isinf(representation_matrix_svd)] = 0

    return representation_matrix_svd
    


#generate representation matrix in case that the matrix is factorised with autoencoder    
representation_matrix_autoencoder = np.zeros((nodes_num, step*dim))
def GetRepMat_autoencoder():
    for i in range(step):
        probTranMat = GetProbTranMat(k_transition_matrix[:,:,i])

        Rk = test_dA(learning_rate=0.1, training_epochs=500,
            probTranMat = probTranMat, n_visible=nodes_num, n_hidden=dim)
        Rk = np.asarray(Rk.eval(), dtype = np.float32)
        
        Rk = np.divide(Rk, np.repeat(np.sqrt(np.sum(np.power(Rk, 2), axis=1)).reshape(-1,1), dim, axis=1))
        representation_matrix_autoencoder[:, dim*(i):dim*(i+1)] = Rk;
        
    #for nodes whose feature factor consists Inf and NaN because of normalisation
    representation_matrix_autoencoder[np.isnan(representation_matrix_autoencoder)] = 0
    representation_matrix_autoencoder[np.isinf(representation_matrix_autoencoder)] = 0

    return representation_matrix_autoencoder
 
        
representation_matrix_svd = GetRepMat_svd()
print("\nIn case of factorisation of matrix with SVD, the representation matrix is: ")
print(representation_matrix_svd)


print("\nTraining autoencoder ... ... ")
start_time = time()
representation_matrix_autoencoder = GetRepMat_autoencoder()
#Get the information about training process
end_time = time()
sec=end_time-start_time
print("The overall Training time: "+str(strftime("%H:%M:%S",gmtime(sec))))

print("\nIn case of factorisation of matrix with autoencoder, the representation matrix is: ")
print(representation_matrix_autoencoder)


#clustering with K-means algorithm
print("\nClustering with Kmeans ... ... ")
centroids, C = kMeans(representation_matrix_autoencoder, K = num_clusters)
print("\nThe indices of clusters, to which each node belongs.")
print(C)


#visualise the clustering result using TSNE tool
model = TSNE(n_components=2)
Rep_2dim = model.fit_transform(representation_matrix_autoencoder)
clusters = np.asarray([Rep_2dim[C == k]  for k in range(num_clusters) if np.any(Rep_2dim[C == k])])

plt.scatter(clusters[0][:, 0], clusters[0][:, 1], c='red')
plt.scatter(clusters[3][:, 0], clusters[3][:, 1], c='green')
plt.scatter(clusters[4][:, 0], clusters[4][:, 1], c='blue')
plt.show()



