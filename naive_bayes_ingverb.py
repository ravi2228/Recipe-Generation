import numpy as np
from ing_verb_ordering import constants, preproc, most_common, clf_base, evaluation
from ing_verb_ordering import scorer, tagger_base, naive_bayes, hmm, viterbi, bilstm

TRAIN_FILE = constants.TRAIN_FILE
DEV_FILE = constants.DEV_FILE
TEST_FILE = constants.TEST_FILE
START_TAG = constants.START_TAG
END_TAG = constants.END_TAG

all_tags = preproc.get_all_tags(TRAIN_FILE)
all_tags_dev = preproc.get_all_tags(DEV_FILE)
all_tags_tst = preproc.get_all_tags(TEST_FILE)

all_tags = all_tags.union(all_tags_dev)
all_tags = all_tags.union(all_tags_tst)
all_tags.add(START_TAG)
all_tags.add(END_TAG)


add_tagger = lambda words, alltags : ['add' for word in words]

confusion = tagger_base.eval_tagger(add_tagger,'add.preds',all_tags=all_tags)
print (scorer.accuracy(confusion))


theta_mc = most_common.get_most_common_word_weights(TRAIN_FILE)

tagger_mc = tagger_base.make_classifier_tagger(theta_mc)


nb_weights = naive_bayes.get_nb_weights(TRAIN_FILE, .01, all_tags)

# obtaining vocab of words
vocab = set([word for tag,word in nb_weights.keys() if word is not constants.OFFSET])
print (sum(np.exp(nb_weights[('add',word)]) for word in vocab))
print (sum(np.exp(nb_weights[('mince',word)]) for word in vocab))
print (sum(np.exp(nb_weights[('remove',word)]) for word in vocab))

print (nb_weights[('add','baaaaaaaaad')])
print (nb_weights[('add',constants.OFFSET)])
print (nb_weights[('mince',constants.OFFSET)])
print (nb_weights[('remove',constants.OFFSET)])

sum(np.exp(nb_weights[(tag,constants.OFFSET)]) for tag in all_tags)

confusion = tagger_base.eval_tagger(tagger_base.make_classifier_tagger(nb_weights),'nb-simple.preds')
dev_acc = scorer.accuracy(confusion)
print (dev_acc)