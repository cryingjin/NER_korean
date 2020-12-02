import pickle
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans as km
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import random
import numpy as np

def visualize(data,type='tsne'):
    fig = plt.figure(figsize=(12, 8))
    ax = Axes3D(fig)

    sample = random.sample(list(data),2000)
    print(np.array(sample).shape)

    vector = np.array(sample)[:,:128]
    label = np.array(sample)[:,128:]

    if type == 'tsne':
        tsne = TSNE(n_components=3) 
        tsne_results = tsne.fit_transform(vector)
        ax.scatter(tsne_results[:,0], tsne_results[:,1],tsne_results[:,2], s=50,c=label, edgecolors='white')
        ax.set_title('T-SNE of distribution')
        ax.set_xlabel('tsne1')
        ax.set_ylabel('tsne2')
        ax.set_zlabel('tsne3')
        plt.show()


    else:
        pca3 = PCA(n_components=3)
        data_pca3 = pca3.fit_transform(vector)
        ax.scatter(data_pca3[:,0], data_pca3[:,1],data_pca3[:,2], s=50,c=label, edgecolors='white')
        ax.set_title('PCA of Target distribution')
        ax.set_xlabel('pc1')
        ax.set_ylabel('pc2')
        ax.set_zlabel('pc3')
        plt.show()