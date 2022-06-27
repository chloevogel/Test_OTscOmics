
__version__ = 'dev'
#####################################################

import numpy as np
import scipy as sp
import torch
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist

from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score

from tqdm import tqdm

import otscomics

import scanpy as sc
import anndata as ad

import os

######################################################

def load_data(file : str):
    data = pd.read_csv(file, index_col=0)

    # We retrieve the cluster each cell belongs to and group the cells in clusters
    clusters = [name.split('_')[-1] for name in data.columns]
    idx = np.argsort(clusters)

    # We check the different clusters
    names = list(dict.fromkeys(clusters))

    # We keep only the 10_000 more important lines ( = genes) to compute the code faster 
    data = data.iloc[np.argsort(data.std(1))[::-1][:10000]]

    return data, clusters, idx, names


def pearson_distance_matrix(data : pd.DataFrame):
    D_p = cdist(data.T, data.T, metric='correlation')
    D_p /= D_p.max()
    return D_p

def OT_distance_matrix(data : pd.DataFrame) :
    # Compute OT distance matrix

    C = otscomics.cost_matrix(data.to_numpy(np.double), 'cosine')

    # Per-cell normalization (mandatory)
    data_norm = data.to_numpy(np.double)
    data_norm /= data_norm.sum(0)
    # Add a small value to avoid numerical errors
    data_norm += 1e-9
    data_norm /= data_norm.sum(0)

    D_ot, errors = otscomics.OT_distance_matrix(
        data=data_norm, cost=C, eps=.1,
        dtype=torch.double, device='cuda', batch_size = 16
    )

    return D_ot




def hierarchical_clustering(D : np.ndarray, clusters : np.ndarray, n_clusters = -1):
    if n_clusters > 0 :
        clustering = AgglomerativeClustering(n_clusters = n_clusters, affinity = "precomputed", linkage = "complete" )
        n_fin = n_clusters
    else :
        (S_score, n_fin) = (-1,0)
        clustering = AgglomerativeClustering(affinity = "precomputed", linkage = "complete" )
        for n in range(3,26):
            clustering.set_params(n_clusters = n).fit(D)
            S_int = silhouette_score(clustering.labels_.reshape(-1,1),np.array(clusters))
            if S_int > S_score :
                S_score, n_fin = S_int,n
        clustering.set_params(n_clusters = n_fin)

    # Fitting on Euclidian D-Matrix
    clustering.fit(D)

    return clustering.labels_, n_fin


def spectral_clustering(D, clusters) : 
    clustering = SpectralClustering(affinity = "precomputed")
    A = 1 - D/D.max()
    (S_score, n_fin) = (-1,0)
    for n_clusters in range(3,26):
        clustering.set_params(n_clusters = n_clusters)
        clustering.fit(A)
        S_int = silhouette_score(clustering.labels_.reshape(-1,1),np.array(clusters))
        if S_int > S_score :
            S_score, n_fin = S_int,n_clusters

    clustering.set_params(n_clusters = n_fin)
    clustering.fit(A)

    return clustering.labels_, n_fin


def leiden_clustering(D : np.ndarray, clusters : np.ndarray, adata, OT = False) :

    from scanpy.neighbors import _compute_connectivities_umap as conn_umap

    sc.pp.pca(adata)
    sc.pp.neighbors(adata)

    if OT :
        knn_indices, knn_dists, forest = sc.neighbors.compute_neighbors_umap(
        D, n_neighbors=15, metric='precomputed')

        adata.obsp['distances'], adata.obsp['connectivities'] = conn_umap(
            knn_indices,
            knn_dists,
            adata.shape[0],
            15,
        )

    resolutions = np.linspace(.25, 1.5, 20)
    sils, aris, nmis, amis, n_clusters = [], [], [], [], []

    # Iterate through resolutions
    print('Iterating through resolutions...')
    for resolution in tqdm(resolutions):
    
        sc.tl.leiden(adata, resolution=resolution)

        if len(np.unique(adata.obs['leiden'])) > 1:
            sils.append(S_score(D, adata.obs['leiden']))
            aris.append(ARI(clusters, adata.obs['leiden']))
            nmis.append(NMI(clusters, adata.obs['leiden']))
            amis.append(AMI(clusters, adata.obs['leiden']))
            n_clusters.append(len(np.unique(adata.obs['leiden'])))
        else:
            sils.append(-1)
            aris.append(-1)
            amis.append(-1)
            nmis.append(-1)
            n_clusters.append(-1)

    # Max silhouette score
    print('Resolution selected by silhouette score')
    i = np.argmax(sils)

    return (resolutions[i], aris[i], nmis[i], amis[i], n_clusters[i])   



def S_score(D : np.ndarray, clusters : np.ndarray):
    return silhouette_score(D, clusters, metric = "precomputed")

def C_score(D : np.ndarray, clusters : np.ndarray):
    return otscomics.C_index(D, np.array(clusters))


def ARI(clusters : np.ndarray, labels):
    return adjusted_rand_score(clusters, labels)

def NMI(clusters : np.ndarray, labels) :
    return normalized_mutual_info_score(clusters, labels)

def AMI(clusters : np.ndarray, labels):
    return adjusted_mutual_info_score(clusters, labels)


################################################

# Compute everything
import torch

def test_data(file) :
    # Load data
    torch.cuda.empty_cache()
    data, clusters, idx, names = load_data(file)
    print(file + " loaded")

    # Computing distance matrixes
    D_eu = pearson_distance_matrix(data)
    D_ot = OT_distance_matrix(data)

    print("Distance matrixes computed")

    # Saving the distance matrixes
    size = D_eu.shape[0]
    D = np.array([D_eu, D_ot]).reshape((2,size,size))

    # Calculating S and C-scores
    S_eu = S_score(D_eu, clusters)
    S_ot = S_score(D_ot, clusters)
    C_eu = C_score(D_eu, clusters)
    C_ot = C_score(D_ot, clusters)

    S = np.array([S_eu, S_ot]).reshape((2,1))
    C = np.array([C_eu, C_ot]).reshape((2,1))

    # Creating arrays to stock ARIs, NMIs, AMIs and n_clusters
    # The first line corresponds to D-Eu scores
    # The second line corresponds to D-OT scores
    aris = np.zeros((2,3))
    nmis = np.zeros((2,3))
    amis = np.zeros((2,3))
    n_clus = np.zeros((2,3))

    # Hierarchical clustering supervised (column 0)
    n_clusters = len(names)
    cl_eu, n_clus_eu = hierarchical_clustering(D_eu, clusters, n_clusters = n_clusters)
    cl_ot, n_clus_ot = hierarchical_clustering(D_ot, clusters, n_clusters = n_clusters)
    aris[:,0] = ARI(clusters, cl_eu), ARI(clusters, cl_ot)
    nmis[:,0] = NMI(clusters, cl_eu), NMI(clusters, cl_ot)
    amis[:,0] = AMI(clusters, cl_eu), AMI(clusters, cl_ot)
    n_clus[:,0] = n_clus_eu, n_clus_ot

    print("Hierachical cl. supervised finished")

    # Hierarchical clustering unsupervised (column 1)
    cl_eu, n_clus_eu = hierarchical_clustering(D_eu, clusters)
    cl_ot, n_clus_ot = hierarchical_clustering(D_ot, clusters)
    aris[:,1] = ARI(clusters, cl_eu), ARI(clusters, cl_ot)
    nmis[:,1] = NMI(clusters, cl_eu), NMI(clusters, cl_ot)
    amis[:,1] = AMI(clusters, cl_eu), AMI(clusters, cl_ot)
    n_clus[:,1] = n_clus_eu, n_clus_ot

    print("Hierachical cl. unsupervised finished")

    # Spectral clustering (column 2)
    adata = ad.AnnData(data.T)
    adata.obs['cell_line'] = clusters
    cl_eu, n_clus_eu = spectral_clustering(D_eu, clusters)
    cl_ot, n_clus_ot = spectral_clustering(D_ot, clusters)
    aris[:,2] = ARI(clusters, cl_eu), ARI(clusters, cl_ot)
    nmis[:,2] = NMI(clusters, cl_eu), NMI(clusters, cl_ot)
    amis[:,2] = AMI(clusters, cl_eu), AMI(clusters, cl_ot)
    n_clus[:,2] = n_clus_eu, n_clus_ot

    print("Spectral clustering finished")

    return D, S, C, aris, nmis, amis, n_clus





    






