"""
2차 해커톤 한국어 개체명 인식
make embedding vector cluster feature
"""
from utils import *
from sklearn.cluster import DBSCAN
import seaborn as sns
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import collections
# import pyLDAvis
# from kmeans_visualizer import kmeans_to_prepared_data


# load token(형태소 단위)
with open('token/train_token.pickle','rb') as fr:
    train_token = pickle.load(fr)

# load glove embedding vector(형태소)
with open('emb_word_dict_1202 (1).pickle','rb') as fr:
    embedding = pickle.load(fr)

# load 개체명 사전 vector(형태소)
with open('nedict_onehot.pickle','rb') as fr:
    nerdict = pickle.load(fr)



def mk_cluster_feature(token,embedding=None,w2v=True,vector_size=128,km=True,dbscan=False,show=True,one_hot=True,num_cluster=5):
    """
    embedding : 29084 x vector_size(128), np array
    -> only for using extenal embedding vector
    """
    # W2V
    if w2v:
        model = Word2Vec(sentences=token,size=vector_size,window=10,sg=1,min_count=1,workers=-1,iter=20)
        embedding = model.wv.vectors
        print(f"embedding shape :{embedding.shape}") # unique token 29084
    else: # using glove 
        embedding_vector = np.array(list(embedding.values()))
        embedding_key = list(embedding.keys())

    # K-means
    if km:
        kmeans = KMeans(n_clusters=num_cluster,random_state=2020) # target class 갯수만큼 clustering
        kmeans.fit(embedding_vector)
        cluster = kmeans.predict(embedding_vector)
        print(collections.Counter(cluster))
        label = kmeans.labels_
        word_cluster_map = {embedding_key[i]: cluster[i] for i in range(len(embedding_key))}

    # DBSCAN
    if dbscan:
        dbscan_model = DBSCAN(min_samples = 100, eps=0.1)
        cluster = dbscan_model.fit_predict(embedding)
        names = model.wv.index2word
        word_cluster_map = {names[i]: cluster[i] for i in range(len(names))}

    # Viz
    if show:
        visualize(embedding_vector,label,type='pca')
        # visualize(embedding,label,type='tsne')

        for clusters in range(0,6):
            print("\nCluster {}".format(clusters))
            words = []
            for i in np.random.randint(0, 10000, size=100): 
                if( list(word_cluster_map.values())[i] == clusters ):
                    words.append(list(word_cluster_map.keys())[i])
            print(words)

    # mk cluster feature vector
    cluster_feature = []
    ner_feature = []
    pos_feature = []

    if one_hot:
        cluster = to_categorical(cluster)
        
    word_cluster_map2 = {embedding_key[i]: cluster[i] for i in range(len(embedding_key))}

    # 유니크한 각 형태소에 대해 feature 생성 및 합치기
    for w in embedding_key:
        cluster_feature.append(list(word_cluster_map2[str(w)]))
        ner_feature.append(list(ner_dict[str(w)]))
        pos_feature.append(list(pos_dict[str(w)]))

    
    cluster_feature = np.array(cluster_feature)
    ner_feature = np.array(ner_feature)
    pos_feature = np.array(pos_feature)

    print(cluster_feature.shape,ner_feature.shape,pos_feature.shape)

    res = np.concatenate((embedding_vector,cluster_feature,ner_feature,pos_feature),axis=1)

    total_feature_dict = {embedding_key[i] : res[i] for i in range(len(embedding_key))}

    return total_feature_dict

total = mk_cluster_feature(train_token, embedding=embedding,w2v=False,num_cluster=6)
