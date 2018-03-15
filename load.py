import itertools
import re
import numpy as np

def load_data(config):
    """ Loads starter word vectors, and train/dev/test data. """

    #This is where we keep all data
    config.data = {}

    #Loads the starter word vectors
    print "INFO: Loading word embeddings!"
    word_vectors, words = load_embeddings(config.w_dic, config.w_vector)
    words.append(config.unk)
    words.append(config.dig)
    word.append(config.pad)
    ep = np.sqrt(np.divide(3.0, word_vectors.shape[1]))
    unk_vec = np.random.uniform(low=-ep, high=ep, size=(1, word_vectors.shape[1]))
    dig_vec = np.random.uniform(low=-ep, high=ep, size=(1, word_vectors.shape[1]))
    pad_vec = np.zeros(1, word_vectors.shape[1])
    word_vectors = np.append(word_vectors, unk_vec, axis=0)
    word_vectors = np.append(word_vectors, dig_vec, axis=0)
    word_vectors = np.append(word_vectors, pad_vec, axis=0)
    config.w_em_size = word_vectors.shape[1]
    config.data['word_vectors'] = word_vectors

    id_w = dict(enumerate(words))
    w_id = {v:k for k,v in id_w.iteritems()}
    config.data['id_w'] = id_w
    config.data['w_id'] = w_id

    #Loads the starter chars
    print "INFO: Loading characters!"
    _, chars = load_embeddings(config.ch_dic, None)

    chars.append(config.pad)
    config.ch_size = len(chars)
    ep = np.sqrt(np.divide(3.0, config.ch_em_size))
    char_vectors = np.random.uniform(low=-ep, high=ep, size=(config.ch_size, config.ch_em_size))
    char_vectors[config.ch_size-1] = np.zeros(1, config.ch_em_size)
    config.data['char_vectors'] = char_vectors
    id_ch = dict(enumerate(chars))
    ch_id = {v:k for k,v in id_ch.iteritems()}
    config.data['id_ch'] = id_ch
    config.data['ch_id'] = ch_id

    #Loads the tags
    print "INFO: Loading tags!"
    tags = []
    with open(config.tag_dic, 'r') as fd:
        for line in fd.readlines():
            tag = line.strip()
            if len(tag)!=0:
                tags.append(tag)
    tags.append(config.rare)
    tags.append(config.pad)

    id_tag = dict(enumerate(tags))
    tag_id = {v:k for k,v in id_tag.iteritems()}
    config.tag_size = len(tags)
    config.data['id_tag'] = id_tag
    config.data['tag_id'] = tag_id

    if config.mode == 'train':
        #Loads the training set
        print "INFO: Loading training data!"
        load_dataset(config, 'train')

        #Loads the dev set (for tuning hyperparameters)
        print "INFO: Loading dev data!"
        load_dataset(config, 'dev')

        config.data['test'] = None

    elif config.mode == 'test':
        #Loads the test set
        print "INFO: Loading test data!"
        load_dataset(config, 'test')

        config.data['train'] = None
        config.data['dev'] = None

    return
def load_embeddings(vocabfile, vectorfile=None):
    em = None
    if(vectorfile is not None):
        em = np.loadtxt(vectorfile, dtype=np.float32)
    with open(vocabfile) as fd:
        tokens = [line.strip() for line in fd]
    return em, tokens
def load_dataset(config, local_mode):
    if local_mode == 'train':
        f_raw = config.train_raw
        f_ref = config.train_ref

    elif local_mode == 'dev':
        f_raw = config.dev_raw
        f_ref = config.dev_ref

    elif local_mode == 'test':
        f_raw = config.test_raw
        f_ref = None

    sentences = []
    with open(f_raw, 'r') as fd:
        cur = []
        for line in fd:
            line = line.strip()
            #new sentence on blank line
            if (len(line) == 0):
                if len(cur) > 0:
                    sentences.append(cur)
                cur = []
            else: # read in tokens
                cur.append(line)

        # flush running buffer
        if len(cur)!=0: sentences.append(cur)

    if local_mode != 'test':
        labels = []
        with open(f_ref, 'r') as fd:
            cur = []
            for line in fd:
                line = line.strip()
                #new sentence on blank line
                if (len(line) == 0):
                    if len(cur) > 0:
                        labels.append(cur)
                    cur = []
                else: # read in tokens
                    cur.append(line)

            # flush running buffer
            if len(cur)!=0: labels.append(cur)


    Word_X = []
    Cap_X = []
    Char_X = []

    word_x = []
    cap_x = []
    char_x = []
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            word = sentences[i][j]
            word_x.append(canonicalize_word(config, word))
            cap_x.append(capalize_word(word))
            char_x.append(canonicalize_char(config, word))

        Word_X.append(word_x)
        Cap_X.append(cap_x)
        Char_X.append(char_x)
        #reset
        word_x = []
        cap_x = []
        char_x = []

    Y = []
    if local_mode != 'test':
        y = []
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                tag = labels[i][j]
                y.append(canonicalize_tag(config, tag))
            Y.append(y)
            #reset
            y = []


    if len(Y)==0:
        return padding(config, local_mode, Char_X, Cap_X, Word_X)

    return padding(config, local_mode, Char_X, Cap_X, Word_X, Y)
#https://github.com/glample/tagger/blob/master/utils.py
def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags
#https://github.com/glample/tagger/blob/master/utils.py
def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags
#https://github.com/glample/tagger/blob/master/utils.py
def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True
#https://github.com/glample/tagger/blob/master/loader.py
def capalize_word(word):
    """
        Capitalization feature:
        0 = low caps without digits
        1 = all caps without digits
        2 = first letter caps without digits
        3 = one capital (not first letter) without digits
    """
    if word.lower() == word:
	return 0

    elif word.upper() == word:
        return 1

    elif word[0].upper() == word[0]:
    	return 2

    else:
    	return 3
def canonicalize_word(config, word):
    word = word.lower()
    if word in config.data['w_id']:
        return config.data['w_id'][word]
    elif re.search(r'\d', word):
        return config.data['w_id'][config.dig]
    else:
        return config.data['w_id'][config.unk]
def canonicalize_tag(config, tag):
    if tag in config.data['tag_id']:
        return config.data['tag_id'][tag]
    else:
        print "INFO: could not find this tag: ", tag
        print "INFO: replaced it with rare tag!"
        return config.data['tag_id'][config.rare]
def canonicalize_char(config, word):
	word = word.lower()
	lst = []
	for ch in list(word):
		if ch in config.data['ch_id']:
			lst.append(config.data['ch_id'][ch])

	return


    lst
def padding(config, local_mode, char_data, cap_data, word_data, tag_data=None):

    word_X = []
    cap_X = []
    mask_X = []
    char_X = []
    rev_char_X = []
    word_length_X = []
    sentence_length_X = []
    Y = []
    for index in range(len(word_data)):
        sentence = word_data[index]
        if local_mode!='test': tags = tag_data[index]
        caps = cap_data[index]
        sentence_length_X.append(len(sentence))
        j = len(sentence)
        mask_list = [1.0] * j
        if j < config.max_s_len:
            while j < config.max_s_len:
                sentence.append(config.data['w_id'][config.pad])
                if local_mode!='test': tags.append(config.data['tag_id'][config.pad])
                caps.append(0)
                mask_list.append(0.0)
                j += 1
        else:
            sentence = sentence[0:config.max_s_len]
            if local_mode!='test': tags = tags[0:config.max_s_len]
            caps = caps[0:config.max_s_len]
            mask_list = mask_list[0:config.max_s_len]

        cap_X.append(caps)
        word_X.append(sentence)
        mask_X.append(mask_list)
        if local_mode!='test': Y.append(tags)

    for index in range(len(char_data)):
        sentence = char_data[index]
        pad_list = [config.data['ch_id'][config.pad]] * config.max_w_len
        length = []
        new_sentence = []
        rev_new_sentence = []
        for k in range(len(sentence)):
            word = sentence[k]
            rev_word = list(reversed(word))
            length.append(len(word))
            j = len(word)
            if j < config.max_w_len:
                while j < config.max_w_len:
                    word.append(config.data['ch_id'][config.pad])
                    rev_word.append(config.data['ch_id'][config.pad])
                    j += 1
            else:
                word = word[0:config.max_w_len]
                rev_word = list(reversed(word))


            new_sentence.append(word)
            rev_new_sentence.append(rev_word)

        j = len(new_sentence)
        if j < config.max_s_len:
            while j < config.max_s_len:
                new_sentence.append(pad_list)
                rev_new_sentence.append(pad_list)
                length.append(0)
                j += 1
        else:
            new_sentence = new_sentence[0:config.max_s_len]
            rev_new_sentence = rev_new_sentence[0:config.max_s_len]
            length = length[0:config.max_s_len]

        word_length_X.append(length)
        char_X.append(new_sentence)
        rev_char_X.append(rev_new_sentence)

    data = {
            'ch_in_d': np.array(char_X),
            'rev_ch_in_d': np.array(rev_char_X),
            'w_len_d': np.array(word_length_X),
            'w_cap_d': np.array(cap_X),
            'w_in_d': np.array(word_X),
            'w_mask_d': np.array(mask_X),
            's_len_d': np.array(sentence_length_X)
            }

    if tag_data is not None:
        data['tag_d'] = np.array(Y)
    else:
        data['tag_d'] = None

    config.data[local_mode] = data
    return
def data_iterator(config, local_mode, shuffle, total_steps):

    #static batch size
    sb_size = config.batch_size
    if shuffle:
    	steps = np.random.permutation(total_steps).tolist()
    else:
    	steps = range(total_steps)

    out_dic = {}
    for step in steps:
        # Create the batch by selecting up to batch_size elements
        bs = step * sb_size
        out_dic['ch_in_b'] = config.data[local_mode]['ch_in_d'][bs:bs + sb_size][:][:]
        out_dic['rev_ch_in_b'] = config.data[local_mode]['rev_ch_in_d'][bs:bs + sb_size][:][:]
        out_dic['w_len_b'] = config.data[local_mode]['w_len_d'][bs:bs + sb_size][:]
        out_dic['w_cap_b'] = config.data[local_mode]['w_cap_d'][bs:bs + sb_size][:]
        out_dic['w_in_b'] = config.data[local_mode]['w_in_d'][bs:bs + sb_size][:]
        out_dic['w_mask_b'] = config.data[local_mode]['w_mask_d'][bs:bs + sb_size][:]
        out_dic['s_len_b'] = config.data[local_mode]['s_len_d'][bs:bs + sb_size]

        if local_mode!='test':
            out_dic['tag_b'] = config.data[local_mode]['tag_d'][bs:bs + sb_size][:]
        else:
            out_dic['tag_b'] = None
        ###
        yield out_dic
