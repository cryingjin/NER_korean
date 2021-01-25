# AI Hackathon #2, Konkuk Univ.

## Installation

```shell script
$ git clone https://github.com/cryingjin/NER_korean.git
$ pip install -r requirements.txt
```

## Execute

### To Train Model
```shell script
$ python main.py --mode train [--args value]
```

### To Test Model
```shell script
$ python main.py --mode test [--args value]
```

### Options

- `file_path`: 데이터셋 파일의 경로
- `file_name`: 데이터셋 파일 이름
- `test_step`: 스텝마다 Validation Loss 를 확인 가능. 기본 5
- `num_epochs`: epoch 수. 기본 100
- `batch_size`: batch 크기. 기본 64
- `learning_rate`: 학습율. 기본 0.001
- `embedding_size`: 워드 임베딩 차원
- `hidden_size`: LSTM/GRU 은닉층 차원

## View Console
(WIP)



### Moong

go to gazet_util/test.ipynb

루트 디렉토리 : confing["root_dir"] : "../"  
개체명 사전 경로 : config["gazet_file"] : "data/gazette.txt"  
한 문장의 최대 형태소 개수 : config["max_morphs"] 일단 10으로 설정  
