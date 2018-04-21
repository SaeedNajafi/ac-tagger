import sys
import codecs

reload(sys)
sys.setdefaultencoding('utf-8')

#Used for NER
def merge(ref_file, pred_file, new_file):
    merge_file = codecs.open(new_file, 'w', 'utf-8')
    ref_lines = codecs.open(ref_file, 'r', 'utf-8').readlines()
    pred_lines = codecs.open(pred_file, 'r', 'utf-8').readlines()

    if len(ref_lines)!=len(pred_lines):
        print "INFO: Wrong number of lines in reference and prediction files"
        exit()

    for index in range(len(ref_lines)):
        ref_line = ref_lines[index].strip()
        pred_line = pred_lines[index].strip()
        if len(ref_line)!=0 and len(pred_line)!=0:
            Gtag = ref_line
            word, tag = pred_line.split('\t')
            merge_file.write(word + '\t' + Gtag + '\t' + tag + '\n')
        else:
            merge_file.write('\n')

    return

merge(sys.argv[1], sys.argv[2], sys.argv[3])
