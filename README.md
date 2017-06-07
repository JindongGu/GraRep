# Graph_Representation_Learning_with_gloabal_structural_Information
This is a python implementation of the paper "GraRep: Learning Graph Representations with Global Structural Information". 
This project improves the implementation of the paper, applies the idea on a new datasets(Facebook Social Circles),
implements a K-means algorithm, does clustering on the datasets and visulises the result using TSNE tool.

1. The original implementation needs a preprocessing of filering the isolated nodes. However, the isolated nodes can be necessary in some application.
So, This implementation takes isolated nodes into consideration. With changes as following:
remove the isolated nodes in preprocessing step

2. The original paper factorises the matrix using SVD to prove that the learned ditributted learning is improved by the global structural information.
This project generates a distributted representation using Autoencoder instead of Matrix Factorization.

3. This project applies the improved implementation to the datasets (Facebook Social Circles). The learned distribution is used to clustering.
The applied Clustring algorithm is K-means, in which the cosine distance function is used. This project also visulise the culstering result using TSNE tool.
