from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


def train(model, train_data, test_data, num_epochs=20):
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    accuracy_list = []

    for epoch in range(num_epochs):
        model.train()
        losses = []

        for step, batch in enumerate(train_dataloader):
            source = batch['source']
            target = batch['target']

            loss = model.forward(source, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 50 == 0:
                print(
                    '{} step processed.. current loss : {}'.format(
                        step + 1,
                        loss.data.item()
                    )
                )
            losses.append(loss.data.item())

        print('Average Loss : {}'.format(np.mean(losses)))

        torch.save(model, 'output/savepoint.model')
        # do_test(model, test_data, idx2tag)


def test(config):
    # 모델 객체 생성
    model = RNN_CRF(config).cuda()
    # 단어 딕셔너리 생성
    word2idx, idx2word = load_vocab(config['word_vocab_file'])
    tag2idx, idx2tag = load_vocab(config['tag_vocab_file'])

    # 저장된 가중치 Load
    model.load_state_dict(torch.load(os.path.join(
        config['output_dir_path'], config['trained_model_name'])))

    # 데이터 Load
    test_input_features, test_tags = load_data(
        config, config['dev_file'], word2idx, tag2idx)

    # 불러온 데이터를 TensorDataset 객체로 변환
    test_features = TensorDataset(test_input_features, test_tags)
    test_data = DataLoader(
        test_features, shuffle=False, batch_size=config['batch_size'])
    # 평가 함수 호출
    do_test(model, test_data, idx2tag)


def do_test(model, test_data, idx2tag):
    model.eval()

    predicts, answers = [], []

    for step, batch in enumerate(test_data):
        source = batch['source']
        target = batch['target']

        # 예측 라벨 출력
        output = model(source)

        # 성능 평가를 위해 예측 값과 정답 값 리스트에 저장
        for idx, answer in enumerate(tensor2list(labels)):
            answers.extend([idx2tag[e].replace(
                '_', '-') for e in answer if idx2tag[e] != '<SP>' and idx2tag[e] != '<PAD>'])
            predicts.extend([idx2tag[e].replace('_', '-') for i, e in enumerate(
                output[idx]) if idx2tag[answer[i]] != '<SP>' and idx2tag[answer[i]] != '<PAD>'])

    # 성능 평가
    print(classification_report(answers, predicts))
