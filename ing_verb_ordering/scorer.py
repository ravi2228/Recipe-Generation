import codecs
import sys
from collections import defaultdict
# import preproc
# import gtnlplib.constants
# import tagger_base
#scoring script for text classification. 
#first argument should be a key file, containing space-separated filename and label
#second argument should be a response file, containing just the predicted label

def main():
    key = sys.argv[1]
    response = sys.argv[2]
    counts = get_confusion(key,response)
    print_score_message(counts)


def get_confusion(keyfilename, responsefilename):
    """Calculate the confusion matrix

    Parameters:
    keyfilename -- Filename containing correct labels
    responsefilename -- Filename containing produced labels

    Returns:
    counts -- dict of counts of (true_label, pred_label) occurences  
    """
    counts = defaultdict(int)
    with codecs.open(keyfilename,encoding='utf8') as keyfile:
        with open(responsefilename,'r') as resfile:
            for key_line in keyfile:
                if len(key_line.rstrip()) == 0:
                    resfile.readline()
                else:
                    ing_verbs = key_line.strip().split(',')
                    for ing_verb in ing_verbs[1:]:
                        ing_verb_split = ing_verb.split('::')
                        if len(ing_verb_split) == 2:
                            key_tag = ing_verb_split[1].strip()
                            res_line = resfile.readline().rstrip()
                            res_tag = res_line.rstrip()
                            counts[tuple((key_tag,res_tag))] += 1
    return(counts)

def accuracy(counts):
    """Calculate the accuracy from a confusion matrix"""
    return sum([y for x,y in counts.items() if x[0] == x[1]]) / float(sum(counts.values()))

def print_score_message(counts):
    true_pos = 0
    total = 0

    keyclasses = set([x[0] for x in counts.keys()])
    resclasses = set([x[1] for x in counts.keys()])
    print ("%d classes in key: %s" % (len(keyclasses),keyclasses))
    print ("%d classes in response: %s" % (len(resclasses),resclasses))
    print ("confusion matrix")
    print ("key\\response:\t"+"\t".join(resclasses))
    for i,keyclass in enumerate(keyclasses):
        print (keyclass+"\t\t")
        for j,resclass in enumerate(resclasses):
            c = counts[tuple((keyclass,resclass))]
            print ("{}\t".format(c))
            total += float(c)
            if resclass==keyclass:
                true_pos += float(c)
        print ("")
    print ("----------------")
    print ("accuracy: %.4f = %d/%d\n" % (true_pos / total, true_pos,total))


if __name__ == "__main__":
    # START_TAG = gtnlplib.constants.START_TAG
    # END_TAG = gtnlplib.constants.END_TAG
    # all_tags = preproc.get_all_tags( gtnlplib.constants.TRAIN_FILE)
    # all_tags.add(START_TAG)
    # all_tags.add(END_TAG)
    # noun_tagger = lambda words, alltags: ['add' for word in words]
    # confusion = tagger_base.eval_tagger(noun_tagger, 'add.preds', all_tags=all_tags)
    # print(accuracy(confusion))
    main()

