# ready
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
np.set_printoptions(threshold=sys.maxsize)
import os
import tqdm
import warnings
warnings.filterwarnings(action='ignore')
import pickle
# visualization
from matplotlib import pyplot as plt
plt.style.use('seaborn')
import seaborn as sns
%matplotlib inline
from IPython.display import display
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import re
# Font
import matplotlib.font_manager as fm
fm._rebuild()
plt.rc('font', family='NanumGothic')
plt.rc('axes', unicode_minus=False)
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
# kras
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

# sklearn
from sklearn.model_selection import train_test_split,KFold
#from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import gensim
from gensim import corpora
import pickle
import random
from konlpy.tag import Okt
import nltk



# load data
train_data = pd.read_csv('ratings_train.txt', header = 0, delimiter = '\t', quoting = 3)
test_data = pd.read_csv('ratings_test.txt', header = 0, delimiter = '\t', quoting = 3)

# 전처리
train_data.drop_duplicates(subset = ['document'], inplace=True)
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
train_data['document'].replace('', np.nan, inplace=True)
train_data = train_data.dropna(how='any')
train_data

test_data.drop_duplicates(subset = ['document'], inplace=True)
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
test_data['document'].replace('', np.nan, inplace=True)
test_data = test_data.dropna(how='any')
test_data


# 불용어 처리 및 tokenizing
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다','에서','되','그','수','나','것','하','있','보','주','아니','등','같','때','년','가','한','지','오','말','일']

okt=Okt()  
def tknz(sentence):
    s = okt.pos(sentence)
    x = []
    for w in s:
        if w[1] == 'Josa' or w[1] == 'Punctuation' or w[1] == 'Number' or w[1] == 'Modifer' or w[1] == 'Eomi':
            continue
        else: x.append(w[0])
    return x

tokens = []
for s in train_data["document"]:
    x = tknz(str(s))
    tem = [word for word in x if not word in stopwords] 
    tokens.append(tem)

## Doc2vec
from collections import namedtuple
TaggedDocument = namedtuple('TaggedDocument', 'words tags')





# tokenized doc load
with open('train_label_moong.pickle','rb') as fr:
    train_label = pickle.load(fr)

with open('train_moong.pickle','rb') as fr:
    train_token = pickle.load(fr)

with open('train_label_moong.pickle','rb') as fr:
    test_label = pickle.load(fr)

with open('train_moong.pickle','rb') as fr:
    test_token = pickle.load(fr)


train_label.reset_index(drop=True,inplace=True)
label = pd.DataFrame(train_label)

neg_train_idx = label[label['label']==0].index
pos_train_idx = label[label['label']==1].index

neg_train = []
for i in neg_train_idx:
    neg_train.append(train_token[i])

pos_train = []
for i in pos_train_idx:
    pos_train.append(train_token[i])

total_train = pos_train+neg_train

def mk_join_doc(corpus):
    doc = []
    for w in corpus:
        tem = " ".join(w)
        doc.append(tem)
    return doc

pos_doc = mk_join_doc(pos_train)
neg_doc = mk_join_doc(neg_train)
total_train_doc = mk_join_doc(total_train)

unique_word_pos = list(set(w for do in pos_total for w in do))
len(unique_word_pos)
unique_word_neg = list(set(w for do in neg_total for w in do))
len(unique_word_neg)
unique_word_total = list(set(w for do in total_train for w in do))
len(unique_word_total)

# for LDA
pos_train_tag = []
for pos in pos_train:
    tem = pos + ['긍정']
    pos_train_tag.append(tem)

neg_train_tag = []
for neg in neg_train:
    tem = neg + ['부정']
    neg_train_tag.append(tem)

total_train_tag = pos_train_tag + neg_train_tag
random.shuffle(total_train_tag)



# TF-idf
tfidfv = TfidfVectorizer(max_features=3000).fit(pos_doc)
score = {key : value for key, value in zip(sorted(tfidfv.vocabulary_, key = lambda x : tfidfv.vocabulary_[x]), tfidfv.transform(pos_doc).toarray().sum(axis=0))}
top_pos = sorted(score, key = lambda x : score[x],reverse=True)[:350]
# print(top_pos)

tfidfv = TfidfVectorizer(max_features=3000).fit(neg_doc)
score = {key : value for key, value in zip(sorted(tfidfv.vocabulary_, key = lambda x : tfidfv.vocabulary_[x]), tfidfv.transform(neg_doc).toarray().sum(axis=0))}
top_neg = sorted(score, key = lambda x : score[x],reverse=True)[:350]
# print(top_neg)


vocab = top_pos+top_neg

vocab = list(set(vocab))

with open('vocab_list.pickle','wb') as f:
    pickle.dump(vocab,f)



# LDA
dictionary = corpora.Dictionary(total_train_tag)
corpus = [dictionary.doc2bow(text) for text in total_train_tag]
print(dictionary[2])


NUM_TOPICS = 2 #20개의 토픽, k=20
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=20)
topics = ldamodel.print_topics(num_words=500)
for topic in topics:
    print(topic)

hangul = re.compile('[^ ㄱ-ㅣ가-힣]+') # 한글과 띄어쓰기를 제외한 모든 글자
result = hangul.sub('', str(topics[1])) # 한글과 띄어쓰기를 제외한 모든 부분을 제거

topics

# chi_squre 
def chi_stat(t):
    a = 0
    b = 0

    for j in range(len(neg_total)):
        d = neg_total[j]
        a += t in d
        c = len(neg_total) - a

    for k in range(len(pos_total)):
        d = pos_total[k]
        b += t in d
        d = len(pos_total) - b
    
    res = ((a+b+c+d) * ((a*d - c*b)**2))/((a+c) * (b+d) * (a+b) * (c+d))

    return res

res_dict = {}

for w in unique_word_total:
    stat = chi_stat(w)
    res_dict[w] = stat


# 카이제곱 통계량 상위 250개 뽑아서 그 중 두 글자 초과인 단어들만 저장(한 글자나 두 글자는 중요도가 떨어질 것으로 판단)
top250 = sorted(res_dict, key = lambda x : res_dict[x],reverse=True)[:250]
top_250_over2 = [w for w in top250 if len(w) > 2]

len(top_250_over2) # 214개 단어사전

# 위의 단어사전으로 count vector 구성
vector = CountVectorizer()
vector.fit(vocab)
doc_vector = vector.transform(total_train_doc).toarray()

doc_vector.shape

classifier = MultinomialNB()
X_train,X_test,y_train,y_test = train_test_split(doc_vector,label['label'])

classifier.fit(X_train, y_train)
# doc_vector_test = vector.transform(test['document']).toarray()
predictions_ = classifier.predict(X_test).tolist()
print('Accuracy: %.10f' % accuracy_score(y_test, predictions_))



# 기존의 300차원 vector와 concat
concat_vec = np.concatenate((np.array(indexing_x_data),doc_vector),axis=1)

# 9:1로 split 후 svm 학습, inference

X_train, X_test, y_train, y_test = train_test_split(concat_vec,indexing_y_data,test_size=0.2)

svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)

predict = svm.predict(X_test)

print('Accuracy: %.2f' % accuracy_score(y_test, predict))
for index in range(len(x_data[number_of_train_data:number_of_train_data+5])):
  print()
  print("문장 : ", x_data[index])
  print("정답 : ", index2label[test_y[index]])
  print("모델 출력 : ", index2label[predict[index]])

"""### 정확도가 99%로 기존 모델보다 상승했음을 알 수 있다."""