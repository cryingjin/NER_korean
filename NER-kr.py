"""
2차 해커톤 한국어 개체명 인식
make embedding vector cluster feature
"""
from utils import *
from sklearn.cluster import DBSCAN
import seaborn as sns
from tensorflow.keras.utils import to_categorical
import numpy as np

# load token(형태소 단위)
with open('token/train_token.pickle','rb') as fr:
    train_token = pickle.load(fr)

# load glove embedding vector(형태소)
with open('token/train_token.pickle','rb') as fr:
    train_token = pickle.load(fr)

def mk_cluster_feature(token,embedding=None,w2v=True,vector_size=128,km=True,dbscan=False,show=True,one_hot=True):
    """
    embedding : 29084 x vector_size(128), np array
    -> only for using extenal embedding vector
    """
    # W2V
    if w2v:
        model = Word2Vec(sentences=token,size=vector_size,window=10,sg=1,min_count=1,workers=-1,iter=20)
        embedding = model.wv.vectors
        print(f"embedding shape :{embedding.shape}") # unique token 29084
    else:
        embedding = embedding

    # K-means
    if km:
        kmeans = KMeans(n_clusters=11) # target class 갯수만큼 clustering
        kmeans.fit(embedding)
        cluster = kmeans.predict(embedding)
        names = model.wv.index2word
        word_cluster_map = {names[i]: cluster[i] for i in range(len(names))}

    # DBSCAN
    if dbscan:
        dbscan_model = DBSCAN(min_samples = 100, eps=0.1)
        cluster = dbscan_model.fit_predict(embedding)
        names = model.wv.index2word
        word_cluster_map = {names[i]: cluster[i] for i in range(len(names))}

    cluster = cluster[:,np.newaxis]
    data = np.concatenate((embedding,cluster),axis=1)

    # Viz
    if show:
        visualize(data,type='tsne')
        for cluster in range(0,10):
            print("\nCluster {}".format(cluster))
            words = []
            for i in range(0,100): 
                if( list(word_cluster_map.values())[i] == cluster ):
                    words.append(list(word_cluster_map.keys())[i])
            print(words)

    # mk cluster feature vector
    cluster_feature = []
    if one_hot:
        cluster = to_categorical(cluster)
    for sen in token:
        for w in sen:
            cluster_feature.append(list(word_cluster_map[str(w)]))

    return cluster_feature



tem = mk_cluster_feature(train_token)
