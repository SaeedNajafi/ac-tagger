class Configuration(object):
    """Model hyperparams and data information"""
    w_rnn_units = 256
    ch_rnn_units = 32
    ch_em_size = 32
    tag_em_size = 32
    dec_rnn_units = 256
    dropout = 0.5
    learning_rate = 0.0005
    rl_step_size = 0.00005
    max_gradient_norm = 5.
    max_epochs = 128
    early_stopping = 5
    batch_size = 32
    seed = 111

    #task = 'en_NER'
    task = 'de_NER'
    #task = 'POS'
    #task = 'CCG'

    """path to different files"""
    w_dic = './data/de_embeddings/' + 'ge_word_dic.txt'
    w_vector = './data/de_embeddings/' + 'ge_word_vector.txt'

    ch_dic = './data/de_ner_data/' + 'de.ner.chars'
    tag_dic = './data/de_ner_data/' + 'de.ner.tags'
    train_raw = './data/de_ner_data/' + 'de.ner.train.raw'
    train_ref = './data/de_ner_data/' + 'de.ner.train.ref'
    dev_raw = './data/de_ner_data/' + 'de.ner.dev.raw'
    dev_ref = './data/de_ner_data/' + 'de.ner.dev.ref'


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
