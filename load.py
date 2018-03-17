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

    #Loads the starter word vectors.
    print "INFO: Loading word embeddings!"
    word_vectors = np.loadtxt(cfg.w_vector, dtype=np.float32)
    with open(cfg.w_dic) as fd:
        words = [line.strip() for line in fd]

    #Adding constants to words and word_vectors.
    words.extend([cfg.unk, cfg.dig, cfg.time, cfg.date])

    cfg.w_size = len(words)
    cfg.w_em_size = word_vectors.shape[1]

    #Use xavier-uniform distribution to initialize vectors for constants.
    ep = np.sqrt(np.divide(6.0, cfg.w_em_size))
    temp_vec = np.random.uniform(low=-ep, high=ep, size=(4, cfg.w_em_size))
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
        for ch in list(w):
            if ch not in chars:
                chars.append(ch)

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
    cfg.tag_size = len(tags)
    id_tag = dict(enumerate(tags))
    tag_id = {v:k for k,v in id_tag.iteritems()}
    cfg.data['id_tag'] = id_tag
    cfg.data['tag_id'] = tag_id

    #This is an automatically generated pad id.
    #Everything (word, char or tag) with this id will get a zero vector.
    cfg.pad_id = len(chars) + len(words) + len(tags)
    return

#Utility for NER
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

#Utility for NER
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

#Utility for NER
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

#Utility for NER
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
    word = word.lower()
    lst = []
    for ch in list(word):
        if ch in ch_id:
            lst.append(ch_id[ch])
        else:
            print "INFO: Could not find the following char and ignored it: ", ch

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

    Word_Ids = []
    Cap_Ids = []
    Word_to_Chars = {}
    Tag_Ids = []
    S_Lens = []

    word_ids = []
    cap_ids = []
    tag_ids = []
    for (X, Y) in batch:
        #X is one sentence.
        if len(X)==0 and len(Y)==0: continue

        for x in X:
            word = x
            word_id = process_word(cfg, word)
            word_ids.append(word_id)
            if word_id not in Word_to_Chars:
                Word_to_Chars[word_id] = process_chars(cfg, word)
            cap_ids.append(capalize_word(word))

        #finished one sentence, now add inner lists to parent lists.
        Word_Ids.append(word_ids)
        Cap_Ids.append(cap_ids)
        S_Lens.append(len(X))

        #Reset inner lists for the next sentence.
        word_ids = []
        cap_ids = []

        #Y is the tags sequence for the sentence X.
        for y in Y:
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

    #The processed batch is now a dictionary.
    B = {
        'ch': Char_Ids,
        'w_len': W_Len,
        'w_chs': Word_Chars,
        'w': Word_Ids,
        'w_cap': Cap_Ids,
        's_len': S_Lens
        }

    if hasY:
        B['tag'] = Tag_Ids
    else:
        B['tag'] = None

    pad(cfg, B)

    return B

def pad(cfg, B):
    #Dynamically select max sentence and word length for the current batch.
    cfg.max_s_len = max(B['s_len'])
    cfg.max_w_len = max(B['w_len'])

    #Pad w with pad_id
    for sentence in B['w']:
        pad_lst = [cfg.pad_id] * (cfg.max_s_len-len(sentence))
        sentence.extend(pad_lst)

    #Pad w_cap with pad_id
    for sentence in B['w_cap']:
        pad_lst = [cfg.pad_id] * (cfg.max_s_len-len(sentence))
        sentence.extend(pad_lst)

    #Pad tag with pad_id
    if B['tag'] is not None:
        for sequence in B['tag']:
            pad_lst = [cfg.pad_id] * (cfg.max_s_len-len(sequence))
            sequence.extend(pad_lst)

    #Pad ch with pad_id
    for word in B['ch']:
        pad_lst = [cfg.pad_id] * (cfg.max_w_len-len(word))
        word.extend(pad_lst)

    #Pad w_chs with pad_id
    for sentence in B['w_chs']:
        pad_lst = [cfg.pad_id] * (cfg.max_s_len-len(sentence))
        sentence.extend(pad_lst)

    return
