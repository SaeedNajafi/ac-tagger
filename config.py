class Configuration(object):
    """Model hyperparams and data information"""
    w_rnn_units = 512
    ch_rnn_units = 32
    ch_em_size = 32
    tag_em_size = 128
    dec_rnn_units = 512
    dropout = 0.5
    learning_rate = 0.005
    max_gradient_norm = 5.
    max_epochs = 128
    early_stopping = 10
    batch_size = 32
    seed = 1234

    """path to different files"""
    w_dic = './en_embeddings/' + 'glove.100.dic.txt'
    w_vector = './en_embeddings/' + 'glove.100.vectors.txt'

    tag_dic = './en_ner_data/' + 'en.ner.tags'
    train_raw = './en_ner_data/' + 'ner.train.raw'
    train_ref = './en_ner_data/' + 'ner.train.ref'
    dev_raw = './en_ner_data/' + 'ner.dev.raw'
    dev_ref = './en_ner_data/' + 'ner.dev.ref'


    """ Model Type """
    #Independent prediction of the tags.
    model_type = 'INDP'

    #Conditional Random Field
    #model_type = 'CRF'

    #Decoder RNN trained only with teacher forcing
    #model_type = 'TF-RNN'

    #Decoder RNN trained with scheduled sampling.
    #model_type = 'SS-RNN'

    #Decoder RNN trained with differential scheduled sampling.
    #model_type = 'DS-RNN'

    #Also specify k for decaying the sampling probability in inverse sigmoid schedule.
    #Only for 'SS-RNN' and 'DS-RNN'
    #k=25

    #Decoder RNN trained using REINFORCE with baseline.
    #model_type = 'BR-RNN'

    #Decoder RNN trained with Actor-Critic.
    #model_type = 'AC-RNN'

    #For RL, you need to specify gamma and n-step.
    #gamma = 0.9
    #n_step = 4

    #For inference in decoder RNNs, we have greedy search or beam search.
    #Specify the beam size.
    #search = 'greedy'
    #search = 'beam'
    #beamsize = 10
