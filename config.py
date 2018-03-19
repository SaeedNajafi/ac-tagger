class Configuration(object):
    """Model hyperparams and data information"""
    w_rnn_units = 512
    ch_rnn_units = 32
    ch_em_size = 32
    tag_em_size = 128
    dec_rnn_units = 512
    dropout = 0.5
    learning_rate = 0.0005
    max_gradient_norm = 5.
    max_epochs = 128
    early_stopping = 10

    #Default
    batch_size = 32
    seed = 1234

    model_type = 'INDP'
    #model_type = 'RNN'
    #model_type = 'CRF'
    #model_type = 'S-RNN'
    #model_type = 'DS-RNN'
    #model_type = 'AC-RNN'
    #model_type = 'R-RNN'
    #model_type = 'BR-RNN'

    #search = greedy
    #search = beam
    #search = viterbi
    #beamsize = 12

    """path to different files"""
    w_dic = './en_embeddings/' + 'glove.100.dic.txt'
    w_vector = './en_embeddings/' + 'glove.100.vectors.txt'

    tag_dic = './en_ccg_data/' + 'ccg.tags'
    train_raw = './en_ccg_data/' + 'ccg.train.raw'
    train_ref = './en_ccg_data/' + 'ccg.train.ref'
    dev_raw = './en_ccg_data/' + 'ccg.dev.raw'
    dev_ref = './en_ccg_data/' + 'ccg.dev.ref'
