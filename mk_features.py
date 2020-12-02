"""
2차 해커톤 한국어 개체명 인식
make embedding vector cluster feature
"""
from utils import *
from sklearn.cluster import DBSCAN
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import collections

# load token(형태소 단위)
with open('token/train_token.pickle','rb') as fr:
    train_token = pickle.load(fr)

# load glove embedding vector(형태소)
with open('feature_dict/glove_dict.pickle','rb') as fr:
    embedding = pickle.load(fr)

# load 개체명 사전 vector(형태소)
# with open('nedict_onehot.pickle','rb') as fr:
#     nerdict = pickle.load(fr)

# load pos vector(형태소)
with open('feature_dict/pos_dict.pickle','rb') as fr:
    posdict = pickle.load(fr)


def mk_features_vector(token,embedding,posdict,nerdict,km=True,show=True,num_cluster=5):
    """
    단어 임배딩 벡터, 품사 벡터, 개체명 벡터를 합쳐 clustering 후 
    해당 cluster 변수를 추가해
    최종 형태소 feature vector를 생성한다.

    return : ('형태소':[feature vector(np.array)])
    """
    embedding_vector = np.array(list(embedding.values()))
    embedding_key = list(embedding.keys())
    pos_vector = np.array(list(posdict.values()))
    # ner_vector = np.array(list(nerdict.values()))
    print(f'word embedding vector : {embedding_vector.shape} \n pos vector : {pos_vector.shape} \n ner vector :')
    concat_vector = np.concatenate((embedding_vector,pos_vector),axis=1)

    # K-means
    if km:
        kmeans = KMeans(n_clusters=num_cluster,random_state=2020) # target class 갯수만큼 clustering
        kmeans.fit(concat_vector)
        cluster = kmeans.predict(concat_vector)
        print(collections.Counter(cluster))
        label = kmeans.labels_
    # DBSCAN
    else:
        dbscan_model = DBSCAN(min_samples = 100, eps=0.1)
        cluster = dbscan_model.fit_predict(embedding)
        names = model.wv.index2word
        word_cluster_map = {names[i]: cluster[i] for i in range(len(names))}

    # Visualize
    if show:
        visualize(concat_vector,label,type='pca')
        word_cluster_map = {embedding_key[i]: cluster[i] for i in range(len(embedding_key))}
        for c in range(0,num_cluster):
            print("\nCluster {}".format(c))
            words = []
            for i in np.random.randint(0, 10000, size=100): 
                if( list(word_cluster_map.values())[i] == c ):
                    words.append(list(word_cluster_map.keys())[i])
            print(words)

    # add cluster feature
    cluster = to_categorical(cluster)
    final_vector = np.concatenate((concat_vector,cluster),axis=1)
    print(f'Final feature vector : {final_vector.shape}')

    # mk final feature dict
    total_feature_dict = {embedding_key[i]: final_vector[i] for i in range(len(embedding_key))}

    return total_feature_dict

res = mk_features_vector(train_token,embedding,posdict,None)