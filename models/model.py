import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
import pickle
import math


class RNN_CRF(nn.Module):
    def __init__(self, config, p_dist, idx2word):
        super(RNN_CRF, self).__init__()

        self.config = config
        self.eumjeol_vocab_size = config["word_vocab_size"]
        self.word_embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.number_of_tags = config["number_of_tags"]
        self.number_of_heads = config["number_of_heads"]
        self.NE_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

        self.word_embedding = nn.Embedding(num_embeddings=self.eumjeol_vocab_size,
                                           embedding_dim=self.word_embedding_size,
                                           padding_idx=0)

        self.NE_embedding = torch.nn.Embedding(num_embeddings=config["number_of_tags"],
                                               embedding_dim=config["number_of_tags"],
                                               padding_idx=0)

        self.dropout = nn.Dropout(config["dropout"])

        self.p_dist = p_dist

        self.idx2word = idx2word

        with open("total_glove_emb.pickle", "rb") as fr:
            self.emb = pickle.load(fr)

        with open("pos_dict_test.pickle", "rb") as fr:
            self.pos_onehot = pickle.load(fr)

        self.first_bi_gru = nn.GRU(input_size=128 + 14 + 46,
                                   hidden_size=self.hidden_size,
                                   num_layers=1,
                                   batch_first=False,
                                   bidirectional=True)

        self.multihead_attn = nn.MultiheadAttention(self.hidden_size*2, self.number_of_heads,
                                                    kdim=self.number_of_tags,
                                                    vdim=self.number_of_tags)

        self.second_bi_gru = nn.GRU(input_size=self.hidden_size*4,
                                    hidden_size=self.hidden_size,
                                    num_layers=1,
                                    batch_first=False,
                                    bidirectional=True)
        # CRF layer
        self.crf = CRF(num_tags=self.number_of_tags, batch_first=True)

        # # fully_connected layer를 통하여 출력 크기를 number_of_tags에 맞춰줌
        # # (batch_size, max_length, hidden_size*2) -> (batch_size, max_length, number_of_tags)
        self.hidden2num_tag = nn.Linear(
            in_features=self.hidden_size*2, out_features=self.number_of_tags)

    def scaled_dot_product(self, query, key):
        return torch.matmul(query, key) / math.sqrt(self.number_of_tags)

    def forward(self, inputs, labels=None):

        NE_tensor = []
        for i in range(len(inputs)):
            NE_tensor.append(self.NE_idx)
        NE_tensor = torch.tensor(NE_tensor, dtype=torch.long).cuda()

        p_dist_feature = []
        for batch in inputs:
            seq = []
            for char in batch:
                seq.append(self.p_dist[char])
            p_dist_feature.append(seq)
        p_dist_feature = torch.tensor(p_dist_feature, dtype=torch.float).cuda()

        emb_feature = []
        for batch in inputs:
            seq = []
            for char in batch:
                if self.idx2word[char.item()] in self.emb:
                    seq.append(self.emb[self.idx2word[char.item()]])
                else:
                    x = []
                    for i in range(128):
                        x.append(0)
                    seq.append(x)
            emb_feature.append(seq)
        emb_feature = torch.tensor(emb_feature, dtype=torch.float).cuda()

        pos_feature = []
        for batch in inputs:
            seq = []
            for char in batch:
                if self.idx2word[char.item()] in self.pos_onehot:
                    seq.append(self.pos_onehot[self.idx2word[char.item()]])
                else:
                    x = []
                    for i in range(46):
                        x.append(0)
                    seq.append(x)
            pos_feature.append(seq)
        pos_feature = torch.tensor(pos_feature, dtype=torch.float).cuda()

        p_dist_feature = torch.cat((p_dist_feature, emb_feature), dim=2)
        p_dist_feature = torch.cat((p_dist_feature, pos_feature), dim=2)
        eumjeol_inputs = p_dist_feature.permute(1, 0, 2)

        # eumjeol_inputs = self.word_embedding(inputs).permute(1, 0, 2)
        NE_embedded = self.NE_embedding(NE_tensor).permute(1, 0, 2)

        # eumjeol_inputs = torch.cat((eumjeol_inputs, p_dist_feature.permute(1, 0, 2)), dim=2)
        # eumjeol_inputs = torch.cat((eumjeol_inputs, pos_feature.permute(1, 0, 2)), dim=2)

        encoder_outputs, hidden_states = self.first_bi_gru(eumjeol_inputs)

        attn_output, attn_output_weights = self.multihead_attn(
            encoder_outputs, NE_embedded, NE_embedded)

        decoder_input = torch.cat((encoder_outputs, attn_output), 2)

        decoder_outputs = self.dropout(decoder_input)

        decoder_outputs, decoder_hidden_states = self.second_bi_gru(
            decoder_input)

        decoder_outputs = self.dropout(decoder_outputs)

        decoder_outputs = self.hidden2num_tag(decoder_outputs)

        logits = self.scaled_dot_product(
            decoder_outputs.permute(1, 0, 2), NE_embedded.permute(1, 2, 0))

        # if(labels is not None):
        #     loss = 0
        #     for i in range(len(score_mat)):
        #         loss += F.cross_entropy(score_mat[i], labels[i])
        #     return loss
        # else:
        #     return torch.argmax(F.softmax(score_mat,dim=2), dim=2).tolist()

        if(labels is not None):
            log_likelihood = self.crf(emissions=logits,
                                      tags=labels,
                                      reduction="mean")
            loss = log_likelihood * -1.0

            return loss
        else:
            output = self.crf.decode(emissions=logits)

            return output
