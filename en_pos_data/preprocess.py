
train_file = open('en_pos_train.txt', 'r')
dev_file = open('en_pos_dev.txt', 'r')
test_file = open('en_pos_test.txt', 'r')

train_raw = open('pos.train.raw', 'w')
train_ref = open('pos.train.ref', 'w')

dev_raw = open('pos.dev.raw', 'w')
dev_ref = open('pos.dev.ref', 'w')

test_raw = open('pos.test.raw', 'w')
test_ref = open('pos.test.ref', 'w')

tags_file = open('pos.tags', 'w')
tag_names = []

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

            tags = tag.split('|')
            tag_str = '\t'.join(tags)
            for each in tags:
                if each not in tag_names: tag_names.append(each)
            train_raw.write(word + '\n')
            train_ref.write(tag_str + '\n')
        elif len(line)==0:
            train_raw.write('\n')
            train_ref.write('\n')

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
            tags = tag.split('|')
            tag_str = '\t'.join(tags)
            for each in tags:
                if each not in tag_names: tag_names.append(each)
            dev_raw.write(word + '\n')
            dev_ref.write(tag_str + '\n')
        elif len(line)==0:
            dev_raw.write('\n')
            dev_ref.write('\n')


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

            tags = tag.split('|')
            tag_str = '\t'.join(tags)
            for each in tags:
                if each not in tag_names: tag_names.append(each)
            test_raw.write(word + '\n')
            test_ref.write(tag_str + '\n')
        elif len(line)==0:
            test_raw.write('\n')
            test_ref.write('\n')



for each in tag_names:
    tags_file.write(each+'\n')

tags_file.close()

train_file.close()
dev_file.close()
test_file.close()

train_raw.close()
train_ref.close()

dev_raw.close()
dev_ref.close()

test_raw.close()
test_ref.close()
