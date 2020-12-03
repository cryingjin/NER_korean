def posOnehot(sentence, vocab):
    import numpy as np

    onehot_list = []
    for word in sentence:
        onehot = np.zeros(len(vocab))

        if word in vocab:
            ind = vocab.index(word)
            onehot[ind] += 1 
        elif '+' in word:
            onehot[vocab.index('OTHER')] += 1
        else:
            onehot[vocab.index('UNKNOWN')] += 1
            
        onehot_list.append(onehot)
    return onehot_list

def makePosDict(morph_file, pos_file):
    import pickle
    with open(morph_file, 'rb') as fs:
        total_token = pickle.load(fs)
    with open(pos_file, 'rb') as fs:
        total_pos = pickle.load(fs)

    # pos_vocab 만들기
    pos_vocab = ['NNG', 'NNP', 'NNB', 'NNBC', 'NR', 'NP', 'VV', 'VA', 'VX', 'VCP', 'VCN', 'MM', 'MAG', 'MAJ', 'IC', 'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC', 'EP', 'EF', 'EC', 'ETN', 'ETM', 'XPN', 'XSN', 'XSV', 'XSA', 'XR', 'SF', 'SE', 'SSO', 'SSC', 'SC', 'SY'	, 'SL', 'SH', 'SN', 'OTHER', 'UNKNOWN', '<SP>']
    # pos_onehot 벡터 만들기
    pos_onehot = [posOnehot(pos, pos_vocab) for pos in total_pos]
    # pos dict 만들기
    key_list = list(set([y for x in total_token for y in x]))

    output = {}
    for i, sentence in enumerate(total_token):
        for j, word in enumerate(sentence):
            key = key_list[key_list.index(word)] 
            output[key] = pos_onehot[i][j]
    return output
