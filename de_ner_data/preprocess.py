
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

train_file = open('deu.train.v2.txt', 'r')
dev_file = open('deu.testa.v2.txt', 'r')
test_file = open('deu.testb.v2.txt', 'r')

train_raw = open('ner.train.raw', 'w')
train_ref = open('ner.train.ref', 'w')

dev_raw = open('de.ner.dev.raw', 'w')
dev_ref = open('de.ner.dev.ref', 'w')

test_raw = open('de.ner.test.raw', 'w')
test_ref = open('de.ner.test.ref', 'w')

tags_file = open('de.ner.tags', 'w')
tag_names = []

tags_arr = []
for line in train_file.readlines():
    line = line.strip()
    if line!='-DOCSTART-	O':
        if len(line)!=0:
            try:
                word, tag = line.split('\t')
            except:
                print line
                word = line
                tag = line

            train_raw.write(word + '\n')
            tags_arr.append(tag)
        elif len(line)==0:
            train_raw.write('\n')
            iob2(tags_arr)
            tags = iob_iobes(tags_arr)
            for each in tags:
                if each not in tag_names:
                    tag_names.append(each)

                train_ref.write(each + '\n')
            train_ref.write('\n')
            tags_arr = []
    else:
        tags_arr = []

tags_arr = []
for line in dev_file.readlines():
    line = line.strip()
    if line!='-DOCSTART-	O':
        if len(line)!=0:
            try:
                word, tag = line.split('\t')
            except:
                print line
                word = line
                tag = line

            dev_raw.write(word + '\n')
            tags_arr.append(tag)
        elif len(line)==0:
            dev_raw.write('\n')
            iob2(tags_arr)
            tags = iob_iobes(tags_arr)
            for each in tags:
                if each not in tag_names:
                    tag_names.append(each)

                dev_ref.write(each + '\n')
            dev_ref.write('\n')
            tags_arr = []
    else:
        tags_arr = []

tags_arr = []
for line in test_file.readlines():
    line = line.strip()
    if line!='-DOCSTART-	O':
        if len(line)!=0:
            try:
                word, tag = line.split('\t')
            except:
                print line
                word = line
                tag = line

            test_raw.write(word + '\n')
            tags_arr.append(tag)
        elif len(line)==0:
            test_raw.write('\n')
            iob2(tags_arr)
            tags = iob_iobes(tags_arr)
            for each in tags:
                if each not in tag_names:
                    tag_names.append(each)

                test_ref.write(each + '\n')
            test_ref.write('\n')
            tags_arr = []
    else:
        tags_arr = []

train_file.close()
dev_file.close()
test_file.close()

train_raw.close()
train_ref.close()

dev_raw.close()
dev_ref.close()

test_raw.close()
test_ref.close()

for each in tag_names:
    tags_file.write(each+'\n')

tags_file.close()
