from ing_verb_ordering.constants import OFFSET
from .clf_base import predict
from .evaluation import acc
from .preproc import get_all_tags, load_data
import math
import numpy as np
from collections import defaultdict
import ing_verb_ordering.constants as constants


def get_nb_weights(trainfile, smoothing, allTags):
    """
    estimate_nb function assumes that the labels are one for each document, where as in POS tagging: we have labels for 
    each particular token. So, in order to calculate the emission score weights: P(w|y) for a particular word and a 
    token, we slightly modify the input such that we consider each token and its tag to be a document and a label. 
    The following helper code converts the dataset to token level bag-of-words feature vector and labels. 
    The weights obtained from here will be used later as emission scores for the viterbi tagger.
    
    inputs: train_file: input file to obtain the nb_weights from
    smoothing: value of smoothing for the naive_bayes weights
    
    :returns: nb_weights: naive bayes weights
    """
    token_level_docs=[]
    token_level_tags=[]
    title, X, Y = load_data(trainfile)
    for k in range(len(X)):
        words = X[k]
        tags = Y[k]
        token_level_docs += [{word.lower().strip():1} for word in words]
        token_level_tags += tags
    nb_weights = estimate_nb(token_level_docs, token_level_tags, smoothing, allTags)
    return nb_weights


def get_corpus_counts(x,y,label):
    """
    Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    d = defaultdict(float)
    for i, y_label in enumerate(y):
        if y_label == label:
            for word in x[i].keys():
                if word in d.keys():
                    d[word] += x[i][word]
                else:
                    d[word] = x[i][word]
        else:
            for word in x[i].keys():
                if word not in d.keys():
                    d[word] = 0
    return d
    


def estimate_pxy(x,y,label,smoothing,vocab):
    '''
    Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    '''
    d = defaultdict(lambda: 0)
    bigdoc = get_corpus_counts(x, y, label)
    sum_everything_else = 0
    for w in bigdoc:
        sum_everything_else += (bigdoc[w])
    sum_everything_else += smoothing * len(vocab)
    for word in vocab:
        # count(word, label) = # of occurences of w in bigdoc[c]
        count_word_label = bigdoc[word]
        loglikelihood_word_label = math.log(count_word_label + smoothing) - math.log(
            sum_everything_else)  # - math.log(n_c) #+ math.log(b_x,2)
        d[word] = loglikelihood_word_label
    return d
    


def estimate_nb(x,y,smoothing, allTags):
    """
    estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """
    labels = set(allTags)
    counts = defaultdict(float)
    doc_counts = defaultdict(float)
    weights = defaultdict(float)

    for t_label in labels:
        counts[t_label] += 1.0

    for label in labels:
        corpCounts = get_corpus_counts(x, y, label)
        for word in corpCounts.keys():
            doc_counts[word] += corpCounts[word]

    for label in labels:
        theta_t = estimate_pxy(x, y, label, smoothing, list(doc_counts.keys()))
        for word in doc_counts.keys():
            weights[(label, word)] = theta_t[word]
    for label in labels:
        weights[(label, OFFSET)] = math.log(counts[label] / len(allTags))
    return weights




def find_best_smoother(x_tr,y_tr,x_dv,y_dv,smoothers):
    '''
    find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param smoothers: list of smoothing values
    :returns: best smoothing value
    :rtype: float

    '''
    if len(smoothers) == 0:
        return -1
    theBest = smoothers[0]
    theBestScore = 0
    scores = {}
    for smooth in smoothers:
        dev_correct = 0
        test_weights = estimate_nb(x_tr, y_tr, smooth)
        outter = np.array([predict(x_i, test_weights, y_dv)[0] for x_i in x_dv])
        #outter = clf_base.predict_all(x_dv, test_weights, y_dv)
        test_score = acc(outter, y_dv)
        scores[smooth] = test_score
        if test_score >  theBestScore:
            theBest = smooth
            theBestScore = test_score
            #theBestScore = dev_correct/len(x_dv)
    return theBest, scores
    

if __name__ == '__main__':
    # ## Demo
    START_TAG = constants.START_TAG
    END_TAG = constants.END_TAG

    all_tags = get_all_tags(constants.TRAIN_FILE)
    all_tags_dev = get_all_tags(constants.DEV_FILE)
    all_tags_tst = get_all_tags(constants.TEST_FILE)

    all_tags = all_tags.union(all_tags_dev)
    all_tags = all_tags.union(all_tags_tst)
    all_tags.add(START_TAG)
    all_tags.add(END_TAG)
    nb_weights = get_nb_weights(constants.TRAIN_FILE, .01, all_tags)
    print(sum(np.exp(nb_weights[(tag, constants.OFFSET)]) for tag in all_tags))





