
from model import *
from tqdm import tqdm
import torch
import os
import numpy as np
from torch.utils.data import (DataLoader, TensorDataset)
import torch.optim as optim
from seqeval.metrics import classification_report

root_dir = ""
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
# 파라미터로 입력받은 파일에 저장된 단어 리스트를 딕셔너리 형태로 저장
def load_vocab(f_name):
    vocab_file = open(os.path.join(root_dir, f_name),'r',encoding='utf8')
    print("{} vocab file loading...".format(f_name))

    # default 요소가 저장된 딕셔너리 생성
    symbol2idx, idx2symbol = {"<PAD>":0, "<UNK>":1}, {0:"<PAD>", 1:"<UNK>"}

    # 시작 인덱스 번호 저장
    index = len(symbol2idx)
    for line in tqdm(vocab_file.readlines()):
        symbol = line.strip()
        symbol2idx[symbol] = index
        idx2symbol[index]= symbol
        index+=1

    return symbol2idx, idx2symbol

# 입력 데이터를 고정 길이의 벡터로 표현하기 위한 함수
def convert_data2feature(data, symbol2idx, max_length=None):
    # 고정 길이의 0 벡터 생성
    feature = np.zeros(shape=(max_length), dtype=np.int)
    # 입력 문장을 공백 기준으로 split
    words = data.split()

    for idx, word in enumerate(words[:max_length]):
        if word in symbol2idx.keys():
            feature[idx] = symbol2idx[word]
        else:
            feature[idx] = symbol2idx["<UNK>"]
    return feature
        
# 파라미터로 입력받은 파일로부터 tensor객체 생성
def load_data_(config, f_name, word2idx, tag2idx):
    file = open(os.path.join(root_dir, f_name),'r',encoding='utf8')

    # return할 문장/라벨 리스트 생성
    indexing_inputs, indexing_tags = [], []

    print("{} file loading...".format(f_name))

    # 실제 데이터는 아래와 같은 형태를 가짐
    # 문장 \t 태그
    # 세 종 대 왕 은 <SP> 조 선 의 <SP> 4 대 <SP> 왕 이 야 \t B_PS I_PS I_PS I_PS O <SP> B_LC I_LC O <SP> O O <SP> O O O
    for line in tqdm(file.readlines()):
        try:
            id, sentence, tags = line.strip().split('\t')
        except:
            id, sentence = line.strip().split('\t')
        input_sentence = convert_data2feature(sentence, word2idx, config["max_length"])
        indexing_tag = convert_data2feature(tags, tag2idx, config["max_length"])

        indexing_inputs.append(input_sentence)
        indexing_tags.append(indexing_tag)
    indexing_inputs = torch.tensor(indexing_inputs, dtype=torch.long)
    indexing_tags = torch.tensor(indexing_tags, dtype=torch.long)

    return indexing_inputs, indexing_tags

# 파라미터로 입력받은 파일로부터 tensor객체 생성
def load_data(config, f_name, word2idx, tag2idx):
    file = open(os.path.join(root_dir, f_name),'r',encoding='utf8')

    # return할 문장/라벨 리스트 생성
    indexing_inputs, indexing_tags = [], []

    print("{} file loading...".format(f_name))

    # 실제 데이터는 아래와 같은 형태를 가짐
    # 문장 \t 태그
    # 세 종 대 왕 은 <SP> 조 선 의 <SP> 4 대 <SP> 왕 이 야 \t B_PS I_PS I_PS I_PS O <SP> B_LC I_LC O <SP> O O <SP> O O O
    for line in tqdm(file.readlines()):
        try:
            id, sentence, tags = line.strip().split('\t')
        except:
            id, sentence = line.strip().split('\t')
        input_sentence = convert_data2feature(sentence, word2idx, config["max_length"])
        # indexing_tag = convert_data2feature(tags, tag2idx, config["max_length"])

        indexing_inputs.append(input_sentence)
        # indexing_tags.append(indexing_tag)
    indexing_inputs = torch.tensor(indexing_inputs, dtype=torch.long)
    # indexing_tags = torch.tensor(indexing_tags, dtype=torch.long)

    return indexing_inputs

# tensor 객체를 리스트 형으로 바꾸기 위한 함수
def tensor2list(input_tensor):
    return input_tensor.cpu().detach().numpy().tolist()

def train(config):
    # 모델 객체 생성
    Pr_dist = Prob_dist(config)
    p_dist = Pr_dist.make_P_dist()

    
    # 단어 딕셔너리 생성
    word2idx, idx2word = load_vocab(config["word_vocab_file"])
    tag2idx, idx2tag = load_vocab(config["tag_vocab_file"])

    model = RNN_CRF(config, p_dist, idx2word).cuda()

    # 데이터 Load
    train_input_features, train_tags = load_data(config, config["train_file"], word2idx, tag2idx)
    test_input_features, test_tags = load_data(config, config["dev_file"], word2idx, tag2idx)

    # 불러온 데이터를 TensorDataset 객체로 변환
    train_features = TensorDataset(train_input_features, train_tags)
    train_dataloader = DataLoader(train_features, shuffle=True, batch_size=config["batch_size"])

    test_features = TensorDataset(test_input_features, test_tags)
    test_dataloader = DataLoader(test_features, shuffle=False, batch_size=config["batch_size"])

    # 모델을 학습하기위한 optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    accuracy_list = []
    for epoch in range(config["epoch"]):
        model.train()
        losses = []
        for step, batch in enumerate(train_dataloader):
            # .cuda()를 이용하여 메모리에 업로드
            batch = tuple(t.cuda() for t in batch)
            input_features, labels = batch

            # loss 계산
            loss = model.forward(input_features, labels)

            # 변화도 초기화
            optimizer.zero_grad()

            # loss 값으로부터 모델 내부 각 매개변수에 대하여 gradient 계산
            loss.backward()

            # 모델 내부 각 매개변수 가중치 갱신
            optimizer.step()

            if (step + 1) % 50 == 0:
                print("{} step processed.. current loss : {}, {}".format(step + 1, loss.data.item(), step))
            losses.append(loss.data.item())



        print("{} epoch Average Loss : {}".format(epoch+1, np.mean(losses)))

        # 모델 저장
        # torch.save(model.state_dict(), os.path.join(config["output_dir_path"], "epoch_{}.pt".format(epoch + 1)))
        torch.save(model, "model{}".format(epoch))
        do_test(model, test_dataloader, idx2tag)



def test(config):
    # 모델 객체 생성
    model = torch.load('model17')
    # 단어 딕셔너리 생성
    word2idx, idx2word = load_vocab(config["word_vocab_file"])
    tag2idx, idx2tag = load_vocab(config["tag_vocab_file"])

    # 저장된 가중치 Load
    # model.load_state_dict(torch.load(os.path.join(config["output_dir_path"], config["trained_model_name"])))

    # 데이터 Load
    test_input_features = load_data(config, config["dev_file"], word2idx, tag2idx)

    print(test_input_features.shape)
    # 불러온 데이터를 TensorDataset 객체로 변환
    test_features = TensorDataset(test_input_features)
    test_dataloader = DataLoader(test_features, shuffle=False, batch_size=64)
    # 평가 함수 호출
    return do_test(model, test_dataloader, idx2tag)

def do_test(model, test_dataloader, idx2tag):
    model.eval()
    predicts = []
    for step, batch in enumerate(test_dataloader):
        # .cuda() 함수를 이용하요 메모리에 업로드
        batch = tuple(t.cuda() for t in batch)
        # print(len(batch))
        # 데이터를 각 변수에 저장
        input_features = batch[0]

        # 예측 라벨 출력
        output = model(input_features)
        predicts.append(output)
        # 성능 평가를 위해 예측 값과 정답 값 리스트에 저장
        # for idx, answer in enumerate(tensor2list(labels)):
        #     answers.extend([idx2tag[e].replace("_", "-") for e in answer if idx2tag[e] != "<SP>" and idx2tag[e] != "<PAD>"])
        #     predicts.extend([idx2tag[e].replace("_", "-") for i, e in enumerate(output[idx]) if idx2tag[answer[i]] != "<SP>" and idx2tag[answer[i]] != "<PAD>"] )

    # 성능 평가
    return predicts
    # print(classification_report(answers, predicts))
len_test = [87, 123, 67, 64, 93, 111, 79, 49, 115, 55, 44, 65, 62, 42, 50, 49, 113, 56, 139, 64, 52, 138, 49, 71, 59, 42, 48, 96, 35, 130, 51, 59, 93, 61, 72, 51, 120, 100, 103, 62, 73, 104, 80, 54, 120, 69, 114, 87, 86, 50, 144, 69, 94, 64, 68, 48, 93, 46, 102, 107, 45, 62, 75, 43, 53, 106, 53, 130, 62, 39, 74, 114, 93, 52, 134, 62, 119, 134, 133, 76, 41, 62, 37, 62, 61, 82, 76, 49, 51, 54, 124, 55, 135, 142, 104, 74, 158, 57, 143, 98, 56, 43, 50, 96, 60, 72, 238, 43, 54, 49, 120, 40, 45, 82, 43, 127, 123, 74, 55, 81, 110, 59, 78, 113, 54, 35, 66, 63, 76, 180, 54, 101, 115, 38, 72, 109, 38, 63, 82, 51, 90, 53, 51, 59, 52, 49, 55, 77, 46, 159, 47, 102, 135, 77, 108, 88, 167, 63, 43, 96, 46, 55, 42, 71, 98, 66, 55, 101, 62, 49, 57, 88, 79, 59, 183, 42, 91, 50, 101, 50, 47, 108, 117, 65, 144, 52, 131, 42, 90, 103, 86, 81, 42, 51, 60, 59, 52, 193, 76, 54, 94, 49, 96, 85, 137, 53, 61, 133, 68, 83, 75, 118, 99, 132, 75, 73, 48, 54, 48, 43, 56, 69, 71, 72, 98, 67, 48, 101, 57, 69, 119, 95, 33, 66, 69, 69, 89, 144, 84, 116, 62, 46, 246, 204, 134, 88, 55, 57, 115, 42, 199, 64, 61, 103, 113, 87, 67, 49, 86, 36, 63, 55, 75, 87, 75, 94, 49, 62, 90, 108, 49, 116, 59, 53, 58, 46, 72, 93, 67, 84, 55, 84, 52, 101, 52, 49, 72, 72, 43, 44, 57, 73, 49, 57, 80, 60, 82, 72, 52, 87, 52, 105, 69, 108, 76, 47, 51, 45, 56, 99, 42, 169, 142, 66, 163, 140, 218, 40, 38, 170, 121, 83, 94, 65, 90, 93, 51, 50, 104, 77, 51, 44, 62, 143, 43, 48, 42, 70, 95, 106, 81, 105, 56, 46, 52, 59, 51, 51, 45, 110, 45, 59, 114, 50, 97, 81, 50, 91, 78, 122, 81, 92, 74, 57, 114, 63, 137, 51, 115, 51, 104, 21, 25, 41, 141, 63, 32, 17, 57, 34, 55, 64, 49, 67, 10, 41, 59, 138, 37, 18, 39, 54, 74, 103, 116, 79, 72, 19, 40, 136, 66, 46, 166, 121, 31, 97, 73, 100, 22, 56, 30, 82, 50, 100, 65, 34, 17, 15, 18, 68, 22, 86, 59, 31, 48, 62, 110, 83, 90, 69, 78, 114, 88, 36, 59, 56, 86, 68, 56, 37, 42, 26, 108, 54, 88, 32, 143, 37, 96, 30, 35, 45, 63, 58, 123, 66, 58, 70, 29, 60, 205, 50, 36, 92, 38, 70, 95, 34, 34, 24, 187, 69, 111, 77, 53, 35, 100, 87, 24, 17, 25, 52, 15, 41, 65, 38, 36, 36, 139, 53, 124, 45, 46, 25, 91, 52, 79, 80, 147, 54, 126, 49, 113, 75, 27, 79, 11, 35, 45, 39, 103, 32, 76, 15, 8, 37, 19, 125, 44, 30, 43, 43, 107, 119, 101, 64, 30, 58, 48, 68, 36, 147, 57, 87, 42, 108, 23, 40, 48, 201, 93, 24, 15, 69, 47, 13, 62, 92, 74, 17, 9, 68, 87, 38, 38, 98, 97, 84, 51, 66, 28, 66, 150, 146, 19, 37, 105, 64, 101, 48, 51, 66, 11, 158, 125, 54, 99, 60, 30, 75, 45, 74, 58, 66, 59, 86, 73, 69, 52, 27, 61, 36, 58, 45, 57, 82, 64, 59, 3, 135, 25, 46, 45, 71, 104, 15, 17, 83, 27, 30, 47, 21, 92, 59, 86, 74, 70, 81, 25, 34, 54, 108, 27, 92, 56, 44, 60, 34, 115, 79, 25, 59, 38, 72, 22, 87, 49, 107, 73, 107, 14, 79, 71, 92, 161, 131, 25, 124, 94, 69, 21, 32, 69, 26, 29, 110, 61, 52, 17, 89, 38, 51, 39, 54, 35, 41, 50, 28, 62, 66, 77, 7, 111, 15, 161, 140, 62, 55, 143, 56, 12, 105, 22, 42, 69, 42, 200, 12, 47, 60, 31, 122, 95, 205, 28, 80, 47, 14, 125, 6, 61, 55, 16, 110, 47, 26, 92, 79, 20, 54, 19, 18, 51, 11, 29, 16, 93, 94, 87, 54, 86, 150, 54, 67, 64, 111, 69, 48, 141, 64, 168, 129, 8, 25, 189, 133, 50, 52, 48, 29, 18, 55, 100, 35, 18, 93, 101, 151, 64, 109, 27, 77, 56, 106, 27, 55, 57, 86, 55, 119, 66, 43, 90, 75, 19, 16, 138, 79, 10, 20, 88, 136, 26, 77, 50, 34, 22, 55, 66, 24, 39, 95, 126, 24, 85, 76, 82, 60, 39, 142, 28, 43, 34, 80, 126, 32, 67, 103, 47, 82, 51, 9, 130, 58, 107, 23, 80, 81, 57, 48, 123, 56, 36, 27, 99, 73, 131, 18, 97, 117, 89, 17, 198, 113, 65, 90, 54, 88, 40, 49, 77, 80, 42, 154, 58, 48, 110, 64, 43, 41, 129, 18, 120, 23, 56, 38, 53, 59, 29, 34, 66, 103, 25, 62, 23, 106, 86, 60, 25, 220, 46, 17, 86, 53, 33, 49, 59, 55, 19, 91, 71, 89, 96, 102, 93, 96, 87, 31, 12, 85, 107, 98, 67, 52, 78, 49, 102, 48, 13, 30, 63, 128, 32, 87, 45, 150, 5, 42, 50, 115, 19, 52, 25, 52, 62, 136, 25, 23, 17, 37, 42, 77, 22, 50, 102, 12, 15, 45, 64, 144, 58, 50, 72, 104, 74, 48, 65, 46, 45, 123, 115, 24, 93, 29, 71, 47, 109, 180, 135, 71, 43, 80, 65, 86, 81, 29, 101, 44, 84, 77, 49, 42, 32, 123, 40, 113, 45, 40, 85, 104, 108, 41, 43, 38, 91, 33, 67, 39, 92, 36, 80, 133, 78, 40, 98, 120, 78, 128, 32, 60, 104, 46, 14, 156, 66, 77, 32, 88, 56, 52, 42, 54, 111, 65, 82, 36, 46, 47, 62, 54, 50, 19, 52, 68, 33, 33, 81, 18, 68, 49, 50, 79, 22, 139, 81, 107, 20, 13, 32, 115, 178, 34, 65, 79, 49, 31, 108, 81, 35, 74, 89, 24, 33, 25, 27, 52, 73, 183, 21, 60, 87, 49, 72, 52, 48, 158, 55, 61, 9, 86, 19, 80, 122, 61, 36, 40, 14, 138, 72, 26, 42, 34, 41, 33, 37, 65, 43, 85, 127, 65, 72, 23, 25, 63, 67, 134, 46, 13, 97, 31, 66, 33, 66, 75, 52, 56, 70, 61, 63, 81, 93, 36, 86, 63, 155, 105, 96, 75, 66, 47, 70, 63, 22, 99, 81, 79, 110, 62, 40, 33, 53, 85, 45, 35, 47, 70, 11, 48, 13, 32, 151, 70, 117, 85, 26, 47, 30, 44, 52, 12, 8, 5, 64, 64, 59, 89, 79, 64, 92, 146, 28, 185, 76, 56, 15, 54, 65, 20, 119, 116, 133, 85, 34, 17, 61, 81, 51, 58, 71, 30, 62, 167, 8, 106, 83, 39, 36, 76, 33, 58, 35, 27, 167, 112, 68, 51, 36, 142, 87, 67, 25, 137, 56, 170, 27, 38, 26, 57, 79, 43, 29, 86, 39, 93, 178, 43, 35, 55, 63, 26, 79, 107, 100, 78, 82, 83, 84, 125, 18, 39, 58, 199, 23, 28, 60, 93, 20, 23, 125, 23, 47, 26, 24, 35, 83, 13, 41, 85, 46, 20, 47, 58, 41, 52, 74, 66, 63, 92, 43, 105, 73, 62, 61, 54, 63, 24, 94, 123, 108, 121, 55, 75, 151, 110, 39, 60, 72, 184, 42, 53, 24, 50, 86, 168, 14, 89, 38, 83, 109, 127, 103, 22, 12, 15, 104, 74, 31, 54, 56, 61, 89, 16, 31, 75, 47, 47, 144, 73, 63, 33, 25, 138, 67, 36, 90, 47, 68, 69, 84, 56, 60, 66, 72, 27, 38, 93, 84, 41, 27, 73, 74, 13, 119, 29, 67, 23, 33, 122, 70, 132, 82, 22, 52, 27, 40, 75, 110, 35, 93, 36, 104, 139, 61, 32, 108, 36, 63, 89, 88, 62, 39, 67, 87, 57, 20, 108, 83, 30, 119, 74, 56, 84, 94, 33, 177, 148, 78, 45, 17, 29]

def make_answer(predict, config):
    word2idx, idx2word = load_vocab(config["word_vocab_file"])
    tag2idx, idx2tag = load_vocab(config["tag_vocab_file"])

    test_input_features = load_data(config, config["dev_file"], word2idx, tag2idx)
    
    answers = []
    for idx, sent in enumerate(predict):
        s = []
        for c_dix, char in enumerate(sent):
            s.append(idx2tag[char])
        answers.append(s)
    return answers

def predict(model, data, config):
    tag2idx, idx_2_tag = load_vocab(config["tag_vocab_file"])
    model.eval()

    with open('Team6.pred', 'w', encoding='utf-8'):
        for index, source in zip(data):
            output = model(source)

            predictions = []
            size = len(source)
            for i in range(size):
                predictions.append(idx_2_tag[i])

            fp.write('{}\t{}\n'.format(index, ' '.join(predictions)))