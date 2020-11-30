import argparse
import torch
from models import Seq2seq
from models.model_utils import train, test


def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load data
    train_data = None
    test_data = None
    train_loader = None
    test_loader = None

    model = Seq2seq(
        input_size=config.input_size,
        embedding_size=config.embedding_size,
        hidden_size=config.hidden_size,
        output_size=config.output_size,
    ).to(device)

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
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--mode', type=str, default='train')

    config = parser.parse_args()

    main(config)
