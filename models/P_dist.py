
import numpy as np
import torch
from load import *
from tqdm import tqdm

class Prob_dist():
    def __init__(self, config):
        self.config = config

        self.number_of_chars = config["word_vocab_size"]
        self.number_of_NEs = config["number_of_tags"]

    def get_data(self):
        word2idx, idx2word = load_vocab(self.config["word_vocab_file"])
        tag2idx, idx2tag = load_vocab(self.config["tag_vocab_file"])

        train_input_features, train_tags = load_data(self.config, self.config["train_file"], word2idx, tag2idx)

        return train_input_features, train_tags
            
    def make_P_dist(self):    
        train_input_features, train_tags = self.get_data()

        p_dist = np.zeros(shape=(self.number_of_chars, self.number_of_NEs), dtype=np.float64)

        for sent_idx, sent in tqdm(enumerate(train_input_features)):
            for char_idx, char in enumerate(sent):
                label = train_tags[sent_idx][char_idx]
                p_dist[char][label] += 1
        p_dist = torch.tensor(p_dist)

        return torch.nn.functional.softmax(p_dist, dim=1).tolist()