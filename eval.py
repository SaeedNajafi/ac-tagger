import sys

def accuracy(ref_file, pred_file):
    #1-best tagging accuracy
    ref_lines = open(ref_file, 'r').readlines()
    pred_lines = open(pred_file, 'r').readlines()
    if len(ref_lines)!=len(pred_lines):
        print "INFO: Wrong number of lines in reference and prediction files."
        exit()

    total = 0.0
    correct = 0.0
    for index in range(len(ref_lines)):
        ref_line = ref_lines[index].strip()
        pred_line = pred_lines[index].strip()
        if len(ref_line)!=0 and len(pred_line)!=0:
            Gtag = ref_line
            tag = pred_line.split('\t')[1]
            total += 1
            if Gtag==tag:
                correct += 1

    return float(correct/total) * 100

print str(accuracy(sys.argv[1], sys.argv[2]))
