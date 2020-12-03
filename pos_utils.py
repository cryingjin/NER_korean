def posOnehot(sentence, vocab):
    import numpy as np

    onehot_list = []
    for word in sentence:
        onehot = np.zeros(len(vocab))
        if word in vocab:
            ind = vocab.index(word)
            onehot[ind] += 1 
        else:
            onehot['<UNK>'] += 1
            
        onehot_list.append(onehot)
    return onehot_list

def makePosDict(token, key_list, onehot_vector):
    output = {}
    for i, sentence in enumerate(token):
        for j, word in enumerate(sentence):
            key = key_list[key_list.index(word)]
            output[key] = onehot_vector[i][j]
    return output