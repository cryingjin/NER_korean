import argparse
import torch
from data import NERDataset
from data.utils import load_data, generate_dataframe
from models import Seq2seq
from models.model_utils import train, test
from torch.utils.data import DataLoader


def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load data
    train_df = generate_dataframe(load_data('ner_train.txt'))
    test_df = generate_dataframe(load_data('ner_dev.txt'))

    train_data = NERDataset(train_df)
    test_data = NERDataset(test_df)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)

    model = Seq2seq(input_size=train_data.vocab_size, embedding_size=128, hidden_size=256,
                    output_size=train_data.tag_size, device=device).to(device)

    if config.mode == 'train':
        train(model, train_loader, test_loader, config.num_epochs)
    else:
        test(model, test_loader)


if __name__ == '__main__':
    # batch_size: batch 수
    # input_size: vocab 수
    # 추가로 feature 추가할 때마다 변경해줘야 함
    # hidden_size: LSTM Hidden Size (GRU로 변형해도 무방)
    # output_size: 개체명 태그 수 (B-태그, I-태그, O 전부 포함)

    parser = argparse.ArgumentParser()

    parser.add_argument('--file_path', type=str, default='data')
    parser.add_argument('--file_name', type=str, default='dataset.csv')
    parser.add_argument('--test_step', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--mode', type=str, default='train')

    config = parser.parse_args()

    main(config)
