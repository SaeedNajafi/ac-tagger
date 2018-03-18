class Configuration(object):

    """path to different files"""
    w_dic = './en_embeddings/' + 'glove.100.dic.txt'
    w_vector = './en_embeddings/' + 'glove.100.vectors.txt'

    tag_dic = './ccg_data/' + 'ccg.tags'
    train_raw = './ccg_data/' + 'ccg.train.raw'
    train_ref = './ccg_data/' + 'ccg.train.ref'
    dev_raw = './ccg_data/' + 'ccg.dev.raw'
    dev_ref = './ccg_data/' + 'ccg.dev.ref'



    """Model hyperparams and data information"""
    w_rnn_units = 512
    ch_rnn_units = 32
    ch_em_size = 32
    tag_em_size = 128
    dec_rnn_units = 512
    batch_size = 32
    dropout = 0.5
    learning_rate = 0.0005
    max_gradient_norm = 5.
    max_epochs = 128
    early_stopping = 5
    seed = 1234
    model_type = 'INDP'
    #model_type = 'Seq2Seq'
    #beamsize = 4
