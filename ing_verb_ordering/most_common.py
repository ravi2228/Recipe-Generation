import operator
from collections import defaultdict, Counter
from preproc import load_data
from ing_verb_ordering.constants import OFFSET, START_TAG, END_TAG, UNK, TRAIN_FILE

argmax = lambda x : max(x.items(),key=operator.itemgetter(1))[0]


def get_tag_word_counts(trainfile):
    """
    Produce a Counter of occurences of word for each tag

    Parameters:
    trainfile: -- the filename to be passed as argument to conll_seq_generator
    :returns: -- a default dict of counters, where the keys are tags.
    """
    all_counters = defaultdict(lambda: Counter())
    #(words, tags)
    title, X, Y = load_data(trainfile)
    for i in range(len(X)):
        for j in range(len(X[i])):
            all_counters[Y[i][j]][X[i][j]] += 1
    return all_counters

#'''
#keep
def get_tag_to_ix(input_file):
    """
    creates a dictionary that maps each tag (including the START_TAG and END_TAG to a unique index and vice-versa
    :returns: dict1, dict2
    dict1: maps tag to unique index
    dict2: maps each unique index to its own tag
    """
    tag_to_ix={}
    title, X, Y = load_data(input_file)
    for i in range(len(Y)):
        tag_list = Y[i]
        for tag in tag_list:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
    
    #adding START_TAG and END_TAG
    #if START_TAG not in tag_to_ix:
    #    tag_to_ix[START_TAG] = len(tag_to_ix)
    #if END_TAG not in tag_to_ix:
    #    tag_to_ix[END_TAG] = len(tag_to_ix)
    
    ix_to_tag = {v:k for k,v in tag_to_ix.items()}
    
    return tag_to_ix, ix_to_tag
#'''


#keep 
def get_word_to_ix(input_file, max_size=100000):
    """
    creates a vocab that has the list of most frequent occuring words such that the size of the vocab <=max_size, 
    also adds an UNK token to the Vocab and then creates a dictionary that maps each word to a unique index, 
    :returns: vocab, dict
    vocab: list of words in the vocabulary
    dict: maps word to unique index
    """
    vocab_counter=Counter()
    X, Y = load_data(input_file)
    for i in range(len(X)):
        word_list = X[i]
        for word in word_list:
            vocab_counter[word] += 1
    vocab = [word for word,val in vocab_counter.most_common(max_size-1)]
    vocab.append(UNK)
    
    word_to_ix={}
    ix=0
    for word in vocab:
        word_to_ix[word]=ix
        ix+=1
    
    return vocab, word_to_ix


# keep the below
def get_add_weights():
    """Produce weights dict mapping all words as noun"""
    weights = defaultdict(float)
    weights[('add'),OFFSET] = 1.
    return weights


def get_most_common_word_weights(trainfile):
    """
    Return a set of weights, so that each word is tagged by its most frequent tag in the training file.
    If the word does not appear in the training file, the weights should be set so that the output tag is Noun.
    
    Parameters:
    trainfile: -- training file
    :returns: -- classification weights
    :rtype: -- defaultdict

    """
    weights = defaultdict(float)
    counters = get_tag_word_counts(trainfile)
    max_val = 0
    total = 0
    for tag, cnt in counters.items():
        for word in cnt:
            total += cnt[word]
    for tag, cnt in counters.items():
        tag_val = 0
        for word in cnt:
            weights[(tag, word)] = cnt[word]
            if cnt[word] > max_val:
                max_val = cnt[word]
            tag_val += 1
        weights[(tag, OFFSET)] = tag_val/total
    return weights


def get_tag_trans_counts(trainfile):
    """compute a dict of counters for tag transitions

    :param trainfile: name of file containing training data
    :returns: dict, in which keys are tags, and values are counters of succeeding tags
    :rtype: dict
    """
    tags_appeared = {START_TAG, END_TAG}
    tot_counts = defaultdict(lambda : Counter())
    #counters = get_tag_word_counts(trainfile)
    total_transitions = defaultdict(float)
    #gen = conll_seq_generator(trainfile)
    title, X, Y = load_data(trainfile)
    for k in range(len(X)):
        w = X[k]
        t = Y[k]
        total_transitions[(START_TAG, t[0])] += 1
        for i in range(len(t)-1):
            total_transitions[(t[i], t[i+1])] += 1
        tags_appeared = tags_appeared.union(set(t))
        total_transitions[(t[len(t)-1],END_TAG)] += 1
    for tag_1 in tags_appeared:
        if tag_1 != END_TAG:
            for tag_2 in tags_appeared:
                tot_counts[tag_1][tag_2] = total_transitions[(tag_1, tag_2)]
        # if tag_1 != END_TAG:
        #     for tag_2 in tags_appeared:
        #         if tag_2 != START_TAG: #and tag_2 != END_TAG:

    # for tag_2 in tags_appeared:
    #     tot_counts[END_TAG][tag_2] = total_transitions[(END_TAG, tag_2)]
    return dict(tot_counts)

if __name__ == '__main__':
    counters = get_tag_word_counts(TRAIN_FILE)
    for tag, tag_ctr in counters.items():
        print(tag, tag_ctr.most_common(3))
