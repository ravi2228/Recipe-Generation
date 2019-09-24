import codecs
from ing_verb_ordering import constants
#from gtnlplib.bilstm import prepare_sequence
from collections import Counter
import most_common
#from data_processing.extract_info_1M_np import OneMInfo
import numpy as np
from ing_verb_ordering.constants import (
    TRAIN_FILE_TEMPLATE,
    DEV_FILE_TEMPLATE,
    TEST_FILE_TEMPLATE,
    TRAIN_FILE_TEMPLATE_TITLE,
    DEV_FILE_TEMPLATE_TITLE,
    TEST_FILE_TEMPLATE_TITLE,
    START_TAG,
    END_TAG
)
import random
from nltk import word_tokenize


def load_data(input_file):
    """
    loads the entire data from the file into a list of words and their respective tags
    """
    X = []
    Y = []
    title = ''
    with open(input_file, 'r') as in_file:
        for line in in_file.readlines():
            current_X = []
            current_Y = []
            if line.strip() != '':
                pairs = line.split(',')
                if len(pairs) > 0:
                    title = pairs[0]
                    current_X.append(START_TAG)
                    current_Y.append(START_TAG)
                    for word in word_tokenize(title):
                        current_Y.append(constants.TITLE)
                        current_X.append(word)
                    for pair in pairs[1:]:
                        if pair.strip() != '':
                            ing_verb = pair.split('::')
                            if len(ing_verb) == 2:
                                if ing_verb[0] == '':
                                    print('faulty ing')
                                if ing_verb[1] == '':
                                    print('faulty verb')
                                current_X.append(ing_verb[0].strip().lower())
                                current_Y.append(ing_verb[1].strip().lower())
                if len(current_X) > 2 and len(current_Y) > 2:
                    current_X.append(END_TAG)
                    current_Y.append(END_TAG)
                    X.append(current_X)
                    Y.append(current_Y)
    #print(any([len(x) == 0 for x in X]))
    #print(any([len(y) == 0 for y in Y]))
    return X,Y


def load_input_multiple_batch(start_batch_num, num_batches, include_title=False):
    TITLE = constants.TITLE
    START_TAG = constants.START_TAG
    END_TAG = constants.END_TAG
    all_tags = set()
    for i in range(start_batch_num, start_batch_num + num_batches):
        if include_title:
            all_tags_train = get_all_tags(TRAIN_FILE_TEMPLATE_TITLE.format(i))
            all_tags_dev = get_all_tags(DEV_FILE_TEMPLATE_TITLE.format(i))
            all_tags_tst = get_all_tags(TEST_FILE_TEMPLATE_TITLE.format(i))
        else:
            all_tags_train = get_all_tags(TRAIN_FILE_TEMPLATE.format(i))
            all_tags_dev = get_all_tags(DEV_FILE_TEMPLATE.format(i))
            all_tags_tst = get_all_tags(TEST_FILE_TEMPLATE.format(i))
        all_tags = all_tags.union(all_tags_dev)
        all_tags = all_tags.union(all_tags_tst)
        all_tags = all_tags.union(all_tags_train)

    if include_title:
        all_tags.add(TITLE)
    all_tags.add(START_TAG)
    all_tags.add(END_TAG)
    tag_to_ix = {}
    for tag in all_tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)

    word_to_ix = {}
    vocab = set()
    for i in range(start_batch_num, start_batch_num + num_batches):
        vocab_train, word_to_ix_train = most_common.get_word_to_ix(TRAIN_FILE_TEMPLATE.format(i), 6900)
        vocab_dev, word_to_ix_dev = most_common.get_word_to_ix(DEV_FILE_TEMPLATE.format(i), 6900)
        vocab = vocab.union(vocab_train)
        vocab = vocab.union(vocab_dev)

    for word in vocab:
        word_to_ix[word] = len(word_to_ix)

    X_tr = []
    X_dv = []
    X_te = []
    Y_tr = []
    Y_dv = []
    Y_te = []

    for i in range(start_batch_num, start_batch_num + num_batches):
        if include_title:
            X_tr_i, Y_tr_i = load_data(TRAIN_FILE_TEMPLATE_TITLE.format(i))
            X_dv_i, Y_dv_i = load_data(DEV_FILE_TEMPLATE_TITLE.format(i))
            X_te_i, Y_te_i = load_data(TEST_FILE_TEMPLATE_TITLE.format(i))
        else:
            X_tr_i, Y_tr_i = load_data_no_title(TRAIN_FILE_TEMPLATE.format(i))
            X_dv_i, Y_dv_i = load_data_no_title(DEV_FILE_TEMPLATE.format(i))
            X_te_i, Y_te_i = load_data_no_title(TEST_FILE_TEMPLATE.format(i))
        X_tr.extend(X_tr_i)
        X_dv.extend(X_dv_i)
        X_te.extend(X_te_i)
        Y_tr.extend(Y_tr_i)
        Y_dv.extend(Y_dv_i)
        Y_te.extend(Y_te_i)
    return word_to_ix, vocab, tag_to_ix, all_tags, X_tr, Y_tr, X_dv, Y_dv, X_te, Y_te


def load_data_no_title(input_file):
    """
    loads the entire data from the file into a list of words and their respective tags
    """
    X = []
    Y = []
    title = ''
    with open(input_file, 'r') as in_file:
        for line in in_file.readlines():
            current_X = []
            current_Y = []
            if line.strip() != '':
                pairs = line.split(',')
                if len(pairs) > 0:
                    # title = pairs[0]
                    # for word in word_tokenize(title):
                    #     current_Y.append(constants.TITLE)
                    #     current_X.append(word)
                    current_X.append(START_TAG)
                    current_Y.append(START_TAG)
                    for pair in pairs[1:]:
                        if pair.strip() != '':
                            ing_verb = pair.split('::')
                            if len(ing_verb) == 2:
                                if ing_verb[0] == '':
                                    print('faulty ing')
                                if ing_verb[1] == '':
                                    print('faulty verb')
                                current_X.append(ing_verb[0].strip().lower())
                                current_Y.append(ing_verb[1].strip().lower())
                if len(current_X) > 2 and len(current_Y) > 2:
                    current_X.append(END_TAG)
                    current_Y.append(END_TAG)
                    X.append(current_X)
                    Y.append(current_Y)
    #print(any([len(x) == 0 for x in X]))
    #print(any([len(y) == 0 for y in Y]))
    return X,Y


def load_data_given_verbs(input_file):
    """
    loads the entire data from the file into a list of words and their respective tags
    """
    X = []
    Y = []
    title = ''
    with open(input_file, 'r') as in_file:
        for line in in_file.readlines():
            current_X = []
            current_Y = []
            if line.strip() != '':
                pairs = line.split(',')
                if len(pairs) > 0:
                    title = pairs[0]
                    for word in word_tokenize(title):
                        current_Y.append(constants.TITLE)
                        current_X.append(word)
                    for pair in pairs[1:]:
                        if pair.strip() != '':
                            ing_verb = pair.split('::')
                            if len(ing_verb) == 2:
                                if ing_verb[0] == '':
                                    print('faulty ing')
                                if ing_verb[1] == '':
                                    print('faulty verb')
                                current_X.append(ing_verb[0].strip().lower())
                                #current_X.append(ing_verb[1].strip().lower())
                                #current_Y.append(ing_verb[0].strip().lower())
                                current_Y.append(ing_verb[1].strip().lower())
                if len(current_X) > 1 and len(current_Y) > 1:
                    #random.shuffle(current_X)
                    X.append(current_X)
                    Y.append(current_Y)
    #print(any([len(x) == 0 for x in X]))
    #print(any([len(y) == 0 for y in Y]))
    return X,Y


def get_all_tags(input_file):
    """
    Return unique set of tags in the conll file
    
    Parameters:
    input_file -- the name of the input file
    returns -- a set of all the unique tags occuring in the file
    """
    all_tags = set([])
    with open(input_file, 'r') as in_file:
        for line in in_file.readlines():
            if line.strip() != '':
                pairs = line.split(',')
                for pair in pairs[1:]:
                    if pair.strip() != '':
                        ing_verb = pair.split('::')
                        if len(ing_verb) == 2:
                            ing_verb = pair.split('::')
                            if len(ing_verb) == 2:
                                all_tags.add(ing_verb[1].strip().lower())
    return all_tags


def printTextFile(words_list):
    f = open("words_in_data.txt", "w")
    for words in words_list:
        for word in words:
            f.write(word + " ")
        f.write("\n")
    f.close()


def bag_of_words(text):
    '''
    Count the number of word occurences for each document in the corpus

    :param text: a document, as a single string
    :returns: a Counter for a single document
    :rtype: Counter
    '''
    cnt = Counter()
    for w in text.split():
        cnt[w] += 1
    return cnt


def aggregate_counts(bags_of_words):
    '''
    Aggregate word counts for individual documents into a single bag of words representation

    :param bags_of_words: a list of bags of words as Counters from the bag_of_words method
    :returns: an aggregated bag of words for the whole corpus
    :rtype: Counter
    '''

    counts = Counter()
    for bow in bags_of_words:
        for w in bow.keys():
            counts[w] += bow[w]
            #counts.update(bow)
    return counts





if __name__ == '__main__':
    TRAIN_FILE = constants.TRAIN_FILE
    #load_data(TRAIN_FILE)
    get_all_tags(TRAIN_FILE)
    load_data_given_verbs(TRAIN_FILE)
#     lol = loadGloveModel("Glove6B/glove.6B.50d.txt")
#     print(lol["the"])
    # X, Y = load_data("en-ud-train.conllu")
    # printTextFile(X)
    # bows = []
    # for sent in X:
    #     bow = bag_of_words(" ".join(sent))
    #     bows.append(bow)
    # final_bow = aggregate_counts(bows)
    # f = open("vocab_counts.txt", "w")
    # for word, cnt in final_bow.items():
    #     f.write("{} {}\n".format(word, str(cnt)))
    # f.close()