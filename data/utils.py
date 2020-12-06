import os
import pandas as pd
from konlpy.tag import Mecab
from tqdm import tqdm


def load_data(file_name):
    with open(file_name, 'r', encoding='utf-8') as fp:
        return fp.readlines()


def convert_sentence(sentence):
    return sentence.replace(' ', '').replace('<SP>', ' ')


def tag_pos(sentence):
    syllable = sentence.split(' ')
    converted = convert_sentence(sentence)
    morpher = Mecab()
    morph_pos_list = morpher.pos(converted)
    morphs_list = []

    pos_list = []
    index = 0

    for morph_pos in morph_pos_list:
        morph, pos = morph_pos
        morphs_list.append(morph)
        pos_list.append('B_{}'.format(pos))

        for i in range(1, len(morph)):
            pos_list.append('I_{}'.format(pos))

        index += len(morph)
        if index < len(syllable):
            if syllable[index] == '<SP>':
                morphs_list.append('<SP>')
                pos_list.append('<SP>')
                index += 1

    return morphs_list, ' '.join(pos_list)


def generate_dataframe(raw_data):
    data_with_pos = []
    indices = []

    for raw_datum in raw_data:
        id, sentence, tag = raw_datum.rstrip('\n').split('\t')
        pos_list = tag_pos(sentence)
        data_with_pos.append((sentence, pos_list, tag))
        indices.append(id)

    return pd.DataFrame(data_with_pos, columns=['sentence', 'pos', 'tag'], index=indices)


def load_tag_dict():
    fp = open(os.path.join(os.path.dirname(__file__), 'tag_vocab.txt'), 'r')
    tag_dict = {'<UNK>': 0, '<SP>': 1, '<EOS>': 2, '<PAD>': 3}

    index = 2
    for line in tqdm(fp.readlines()):
        tag = line.strip()
        tag_dict[tag] = index
        index += 1

    return tag_dict
