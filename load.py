import itertools
import re
import numpy as np
from random import shuffle

def load_embeddings(cfg):
    #This is where we will keep embeddings data.
    cfg.data = {}

    #Defining some constants. Change this part if you want.
    cfg.unk = 'uunnkknnuu'
    cfg.dig = 'ddiiggiidd'
    cfg.time = 'ttiimmeemmiitt'
    cfg.date = 'ddaatteettaadd'
    cfg.w_pad = 'ppaaddaapp'

    #Loads the starter word vectors.
    print "INFO: Loading word embeddings!"
    word_vectors = np.loadtxt(cfg.w_vector, dtype=np.float32)
    with open(cfg.w_dic) as fd:
        words = [line.strip() for line in fd]

    #Adding constants to words and word_vectors.
    #Pad should be the last.
    words.extend([cfg.unk, cfg.dig, cfg.time, cfg.date, cfg.w_pad])

    cfg.w_size = len(words)
    cfg.w_em_size = word_vectors.shape[1]

    #Use xavier-uniform distribution to initialize vectors for constants.
    ep = np.sqrt(np.divide(6.0, cfg.w_em_size))
    temp_vec = np.random.uniform(low=-ep, high=ep, size=(5, cfg.w_em_size))
    word_vectors = np.append(word_vectors, temp_vec, axis=0)

    #Map each word to id, and vice versa.
    id_w = dict(enumerate(words))
    w_id = {v:k for k,v in id_w.iteritems()}

    #Save word_vectors and mapping dictionaries.
    cfg.data['w_v'] = word_vectors
    cfg.data['id_w'] = id_w
    cfg.data['w_id'] = w_id

    #Finds chars from words.
    print "INFO: Loading characters!"
    chars = []
    for w in words:
        for ch in list(w.decode('utf8')):
            if ch not in chars:
                chars.append(ch)

    #Pad should be the last.
    cfg.ch_pad = '@'
    chars.append(cfg.ch_pad.decode('utf8'))

    cfg.ch_size = len(chars)
    ep = np.sqrt(np.divide(6.0, cfg.ch_em_size))
    char_vectors = np.random.uniform(
                                    low=-ep,
                                    high=ep,
                                    size=(cfg.ch_size, cfg.ch_em_size)
                                    )
    id_ch = dict(enumerate(chars))
    ch_id = {v:k for k,v in id_ch.iteritems()}
    cfg.data['ch_v'] = char_vectors
    cfg.data['id_ch'] = id_ch
    cfg.data['ch_id'] = ch_id

    #Loads the tags
    print "INFO: Loading tags!"
    tags = []
    with open(cfg.tag_dic, 'r') as fd:
        for line in fd.readlines():
            tag = line.strip()
            if len(tag)!=0: #for empty line!
                tags.append(tag)

    cfg.rare = 'rraarreerraarr' #for rare tags
    tags.append(cfg.rare)

    #Pad should be the last.
    cfg.tag_pad = 'PADPAD'
    tags.append(cfg.tag_pad)

    cfg.tag_size = len(tags)
    id_tag = dict(enumerate(tags))
    tag_id = {v:k for k,v in id_tag.iteritems()}
    cfg.data['id_tag'] = id_tag
    cfg.data['tag_id'] = tag_id

    #This is an automatically generated pad id.
    cfg.w_pad_id = cfg.data['w_id'][cfg.w_pad]
    cfg.ch_pad_id = cfg.data['ch_id'][cfg.ch_pad]
    cfg.tag_pad_id = cfg.data['tag_id'][cfg.tag_pad]
    cfg.cap_pad_id = 4

    return

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

def process_word(cfg, word):
    w_id = cfg.data['w_id']
    word = word.lower()
    if word in w_id:
        return w_id[word]

    #Change this part for more rules to detect dates.
    elif re.search(r'\d{4}-\d{2}-\d{2}', word):
        return w_id[cfg.date]

    #Change this part for more rules to detect times.
    elif re.search(r'\d+:\d+:[\d.]+', word):
        return w_id[cfg.time]

    #Detecting other digits.
    elif re.search(r'\d', word):
        return w_id[cfg.dig]

    return w_id[cfg.unk]

def process_tag(cfg, tag):
    tag_id = cfg.data['tag_id']
    if tag in tag_id:
        return tag_id[tag]

    print "INFO: Could not find the following tag and replaced it with 'rare': ", tag
    return tag_id[cfg.rare]

def process_chars(cfg, word):
    ch_id = cfg.data['ch_id']
    word = word.lower().decode('utf8')
    lst = []
    for ch in list(word):
        if ch in ch_id:
            lst.append(ch_id[ch])
        else:
            lst.append(ch_id[cfg.ch_pad])
            print "INFO: Could not find the following char and replaced it with pad: ", ch

    return lst

def load_data(cfg):
    """ Loads train, dev or test data. """
    #We assume that we cannot read the whole data into memory at once.
    #We do not need the whole data, we read batches of the data.
    #The training data will be shuffled. Pseudo shuffling inside the batch.
    #The dev/test data is not shuffled.

    #static batch size
    sb_size = cfg.batch_size

    #local_mode can have three values 'train', 'dev' and 'test'.
    mode = cfg.local_mode

    if mode == 'train':
        f_raw = cfg.train_raw
        f_ref = cfg.train_ref
        hasY = True

    elif mode == 'dev':
        f_raw = cfg.dev_raw
        f_ref = cfg.dev_ref
        hasY = True

    elif mode == 'test':
        f_raw = cfg.test_raw
        f_ref = None
        hasY = False


    batch = []
    counter = 0
    fd_raw = open(f_raw, 'r')
    if hasY: fd_ref = open(f_ref, 'r')
    x_buffer = []
    y_buffer = []
    for x_line in fd_raw:
        x_line = x_line.strip()
        #we assume ref and raw files have the same number of lines.
        if hasY: y_line = fd_ref.readline().strip()

        #new sentence on blank line
        if (len(x_line) == 0):
            if len(x_buffer) > 0:
                batch.append((x_buffer, y_buffer))
                counter += 1
                if counter==sb_size:
                    yield process_batch(cfg, batch)
                    batch = []
                    counter = 0
            x_buffer = []
            y_buffer = []

        else: # read in tokens
            x_buffer.append(x_line)
            if hasY: y_buffer.append(y_line)

    fd_raw.close()
    if hasY: fd_ref.close()

    #flush running buffer
    if counter!=0 or len(x_buffer)!=0:
        batch.append((x_buffer, y_buffer))
        yield process_batch(cfg, batch)

def process_batch(cfg, batch):
    mode = cfg.local_mode

    #Shuffle the batch only for the training data.
    if mode=='train': shuffle(batch)

    hasY = True
    if mode=='test': hasY = False

    Raw_Words = []
    Word_Ids = []
    Cap_Ids = []
    Word_to_Chars = {}
    Tag_Ids = []
    S_Lens = []

    raw_words = []
    word_ids = []
    cap_ids = []
    tag_ids = []
    for (X, Y) in batch:
        #X is one sentence.
        if len(X)==0 and len(Y)==0: continue

        for x in X:
            word = x
            raw_words.append(word)
            word_id = process_word(cfg, word)
            word_ids.append(word_id)
            if word_id not in Word_to_Chars:
                Word_to_Chars[word_id] = process_chars(cfg, word)
            cap_ids.append(capalize_word(word))

        #finished one sentence, now add inner lists to parent lists.
        Raw_Words.append(raw_words)
        Word_Ids.append(word_ids)
        Cap_Ids.append(cap_ids)
        S_Lens.append(len(X))

        #Reset inner lists for the next sentence.
        raw_words = []
        word_ids = []
        cap_ids = []

        #Y is the tags sequence for the sentence X.
        for y in Y:
            if len(y.split('\t'))>1:
                tag = y.split('\t')[0]
            else:
                tag = y
            tag_ids.append(process_tag(cfg, tag))

        Tag_Ids.append(tag_ids)

        #Reset inner list for the next sequence.
        tag_ids = []


    Char_Ids = []
    W_Len = []
    index = 0
    for word_id, chars in Word_to_Chars.iteritems():
        Char_Ids.append(chars)
        W_Len.append(len(chars))
        Word_to_Chars[word_id] = index
        index += 1

    #Word_Chars has a new id for each word of the sentence in the batch.
    #This new id will map the word to its chars list in Char_Ids.
    #This will be used to map word to its chars.
    Word_Chars = []
    for sentence in Word_Ids:
        word_chars_id = []
        for word_id in sentence:
            word_chars_id.append(Word_to_Chars[word_id])

        Word_Chars.append(word_chars_id)


    #Set dynamic batch size
    d_batch_size = len(S_Lens)

    #Creating reversed char sequences
    Rev_Char_Ids = []
    for i in range(len(Char_Ids)):
        lst = list(reversed(Char_Ids[i]))
        Rev_Char_Ids.append(lst)

    #Creating mask for word sequences
    W_Mask = []
    for each in S_Lens:
        lst = [1.0] * each
        W_Mask.append(lst)

    #The processed batch is now a dictionary.
    B = {
        'ch': Char_Ids,
        'rev_ch': Rev_Char_Ids,
        'w_len': W_Len,
        'w_chs': Word_Chars,
        'w': Word_Ids,
        'w_mask': W_Mask,
        'w_cap': Cap_Ids,
        's_len': S_Lens,
        'raw_w': Raw_Words,
        'd_batch_size': d_batch_size
        }

    if hasY:
        B['tag'] = Tag_Ids
    else:
        B['tag'] = None

    pad(cfg, B)

    return B

def pad(cfg, B):
    #Dynamically select max sentence and word length for the current batch.
    B['max_s_len'] = max(B['s_len'])
    B['max_w_len'] = max(B['w_len'])
    #Pad w with w_pad_id
    for sentence in B['w']:
        pad_lst = [cfg.w_pad_id] * (B['max_s_len']-len(sentence))
        sentence.extend(pad_lst)

    #Pad w with w_pad
    for sentence in B['raw_w']:
        pad_lst = [cfg.w_pad] * (B['max_s_len']-len(sentence))
        sentence.extend(pad_lst)

    #Pad w_cap with cap_pad_id
    for sentence in B['w_cap']:
        pad_lst = [cfg.cap_pad_id] * (B['max_s_len']-len(sentence))
        sentence.extend(pad_lst)

    #Pad tag with tag_pad_id
    if B['tag'] is not None:
        for sequence in B['tag']:
            pad_lst = [cfg.tag_pad_id] * (B['max_s_len']-len(sequence))
            sequence.extend(pad_lst)

    #Pad ch with ch_pad_id
    for word in B['ch']:
        pad_lst = [cfg.ch_pad_id] * (B['max_w_len']-len(word))
        word.extend(pad_lst)

    #Pad rev_ch with ch_pad_id
    for word in B['rev_ch']:
        pad_lst = [cfg.ch_pad_id] * (B['max_w_len']-len(word))
        word.extend(pad_lst)

    #Pad w_chs with 0
    for sentence in B['w_chs']:
        pad_lst = [0] * (B['max_s_len']-len(sentence))
        sentence.extend(pad_lst)

    #Pad w_mask with 0.0
    for sentence in B['w_mask']:
        pad_lst = [0.0] * (B['max_s_len']-len(sentence))
        sentence.extend(pad_lst)

    return
