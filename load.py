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
    ep = np.sqrt(np.divide(3.0, word_vectors.shape[1]))
    unk_vec = np.random.uniform(low=-ep, high=ep, size=(1, word_vectors.shape[1]))
    dig_vec = np.random.uniform(low=-ep, high=ep, size=(1, word_vectors.shape[1]))
    word_vectors = np.append(word_vectors, unk_vec, axis=0)
    word_vectors = np.append(word_vectors, dig_vec, axis=0)
    config.w_em_size = word_vectors.shape[1]
    config.data['word_vectors'] = word_vectors
    config.w_size = len(words)
    id_w = dict(enumerate(words))
    w_id = {v:k for k,v in id_w.iteritems()}
    config.data['id_w'] = id_w
    config.data['w_id'] = w_id

    #Loads the starter chars
    print "INFO: Loading characters!"
    chars = []
    for w in words:
        for ch in list(w):
            if ch not in chars:
                chars.append(ch)

    config.ch_size = len(chars)
    ep = np.sqrt(np.divide(3.0, config.ch_em_size))
    char_vectors = np.random.uniform(low=-ep, high=ep, size=(config.ch_size, config.ch_em_size))
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
    tags.append(config.end)
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

    w_d = []
    w_cap_d = []
    s_len_d = []
    w_ch = {}

    word_arr = []
    cap_arr = []
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            word = sentences[i][j]
            word_arr.append(canonicalize_word(config, word))
            cap_arr.append(capalize_word(word))
            if word not in w_ch:
                w_ch[canonicalize_word(config, word)] = canonicalize_char(config, word)

        w_d.append(word_arr)
        w_cap_d.append(cap_arr)
        s_len_d.append(len(sentences[i]))

        #reset
        word_arr = []
        cap_arr = []

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

    data = {
            'w_ch': w_ch,
            'w_d': w_d,
            'w_cap_d': w_cap_d,
            's_len_d': s_len_d
            }

    if len(Y)!=0:
        data['tag_d'] = Y
    else:
        data['tag_d'] = None

    config.data[local_mode] = data
    return
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

	return lst
def pad(config, out_dic):
    config.max_s_len = max(out_dic['s_len_b'])
    config.max_w_len = max(out_dic['w_len_b'])
    for each in out_dic['w_b']:
        pad_lst = [0] * (config.max_s_len-len(each))
        each.extend(pad_lst)

    for each in out_dic['w_cap_b']:
        pad_lst = [0] * (config.max_s_len-len(each))
        each.extend(pad_lst)

    for each in out_dic['ch_b']:
        pad_lst = [0] * (config.max_w_len-len(each))
        each.extend(pad_lst)

    if out_dic['tag_b'] is not None:
        for each in out_dic['tag_b']:
            pad_lst = [config.data['tag_id'][config.end]] * (config.max_s_len-len(each))
            each.extend(pad_lst)
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
        #Create the batch by selecting up to batch_size elements
        bs = step * sb_size
        out_dic['w_b'] = config.data[local_mode]['w_d'][bs:bs + sb_size][:]
        out_dic['w_cap_b'] = config.data[local_mode]['w_cap_d'][bs:bs + sb_size][:]
        out_dic['s_len_b'] = config.data[local_mode]['s_len_d'][bs:bs + sb_size]

        dic = {}
        for each in out_dic['w_b']:
            for inner_each in each:
                dic[inner_each] = config.data[local_mode]['w_ch'][inner_each]

        #for padding
        if 0 not in dic:
            dic[0] = config.data[local_mode]['w_ch'][0]

        ch_b = []
        ch_lst_b = []
        w_len_b = []
        for each in sorted(dic):
            ch_b.append(dic[each])
            ch_lst_b.append(each)
            w_len_b.append(len(dic[each]))


        out_dic['ch_b'] = ch_b
        index_id = dict(enumerate(ch_lst_b))
        id_index = {v:k for k,v in index_id.iteritems()}
        out_dic['w_id_index'] = id_index
        out_dic['w_len_b'] = w_len_b

        if local_mode!='test':
            out_dic['tag_b'] = config.data[local_mode]['tag_d'][bs:bs + sb_size][:]
        else:
            out_dic['tag_b'] = None

        pad(config, out_dic)
        ###
        yield out_dic
