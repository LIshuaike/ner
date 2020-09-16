class Config():
    train = './data/conll03/train.txt'
    dev = './data/conll03/dev.txt'
    test = './data/conll03/test.txt'
    embed = './data/glove.6B.100d.txt'


class CHAR_LSTM_CRF_Config(Config):
    context_size = 1
    embedding_dim = 100
    char_embedding_dim = 100
    char_output_size = 200
    hidden_size = 150
    use_char = True


config = {'char_lstm_crf': CHAR_LSTM_CRF_Config}