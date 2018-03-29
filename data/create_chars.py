import sys

train_file = open(sys.argv[1], 'r')
chars_file = open(sys.argv[2], 'w')

chars = []

for each in train_file.readlines():
    line = each.strip()
    if len(line)!=0:
        for ch in list(line.split('\t')[0]):
            if ch not in chars:
                chars.append(ch)

for ch in chars:
    chars_file.write(ch + '\n')

chars_file.close()
train_file.close()
