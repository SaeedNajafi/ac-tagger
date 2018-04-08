class Configuration(object):
    """Model hyperparams and data information"""
    w_rnn_units = 200
    ch_rnn_units = 25
    ch_em_size = 25
    tag_em_size = 25
    dec_rnn_units = 200
    dropout = 0.5
    learning_rate = 0.0005
    actor_step_size = 0.5
    max_gradient_norm = 5.
    max_epochs = 128
    early_stopping = 10
    batch_size = 32
    seed = 125

    task = 'en_NER'
    #task = 'de_NER'
    #task = 'POS'
    #task = 'CCG'

    """path to different files"""
    w_dic = './data/en_embeddings/' + 'glove.100.dic.txt'
    w_vector = './data/en_embeddings/' + 'glove.100.vectors.txt'

    ch_dic = './data/en_ner_data/' + 'en.ner.chars'
    tag_dic = './data/en_ner_data/' + 'en.ner.tags'
    train_raw = './data/en_ner_data/' + 'ner.train.raw'
    train_ref = './data/en_ner_data/' + 'ner.train.ref'
    dev_raw = './data/en_ner_data/' + 'ner.dev.raw'
    dev_ref = './data/en_ner_data/' + 'ner.dev.ref'


    """ Model Type """
    #Independent prediction of the tags.
    #model_type = 'INDP'

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
