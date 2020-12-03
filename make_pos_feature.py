import numpy as np
import pickle
from pos_utils import *

with open('./data/train_token.pickle', 'rb') as fs:
    train_token = pickle.load(fs)
with open('./data/dev_token.pickle', 'rb') as fs:
    dev_token = pickle.load(fs)
with open('./data/train_pos.pickle', 'rb') as fs:
    train_pos = pickle.load(fs)
with open('./data/dev_pos.pickle', 'rb') as fs:
    dev_pos = pickle.load(fs)

# pos_vocab 만들기
pos_vocab = ['NNG', 'NNP', 'NNB', 'NNBC', 'NR', 'NP', 'VV', 'VA', 'VX', 'VCP', 'VCN', 'MM', 'MAG', 'MAJ', 'IC', 'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC', 'EP', 'EF', 'EC', 'ETN', 'ETM', 'XPN', 'XSN', 'XSV', 'XSA', 'XR', 'SF', 'SE', 'SSO', 'SSC', 'SC', 'SY'	, 'SL', 'SH', 'SN', '<UNK>', '<SP>']
pos_set_train = set([y for x in train_pos for y in x])
pos_set_dev = set([y for x in dev_pos for y in x])
pos_set = list(pos_set_train | pos_set_dev)

pos_vocab = sorted(list(set(pos_vocab) | set(pos_set)))

# pos_onehot 벡터 만들기
train_onehot = [posOnehot(pos, pos_vocab) for pos in train_pos]
dev_onehot = [posOnehot(pos, pos_vocab) for pos in dev_pos]

# pos dict 만들기
key_list_train = list(set([y for x in train_token for y in x]))
key_list_dev = list(set([y for x in dev_token for y in x]))

train_dict = makePosDict(train_token, key_list_train, train_onehot)
dev_dict = makePosDict(dev_token, key_list_dev, dev_onehot) 