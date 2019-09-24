from ing_verb_ordering.constants import START_TAG, END_TAG, OFFSET, UNK
from .naive_bayes import estimate_pxy
import numpy as np
from collections import defaultdict
import torch
import torch.nn
from torch.autograd import Variable


def compute_transition_weights(trans_counts, smoothing):
    """
    Compute the HMM transition weights, given the counts.
    Don't forget to assign smoothed probabilities to transitions which
    do not appear in the counts.
    
    This will also affect your computation of the denominator.

    :param trans_counts: counts, generated from most_common.get_tag_trans_counts
    :param smoothing: additive smoothing
    :returns: dict of features [(curr_tag,prev_tag)] and weights

    """
    weights = defaultdict(float)
    all_tags = list(trans_counts.keys())+ [END_TAG]
    vocab = all_tags.copy()
    #vocab.remove(START_TAG)
    # x = []
    # y = []
    # for pos in trans_counts.keys():
    #     if pos != START_TAG:
    #         x.append(trans_counts[pos].copy())
    #         y.append(pos)
    x = list(trans_counts.values())
    y = list(trans_counts.keys())

    for tag_2 in all_tags:
        n_weights = estimate_pxy(x, y, tag_2, smoothing, vocab)
        for tag_1, probs in n_weights.items():
            if tag_1 == START_TAG or tag_2 == END_TAG:
                weights[(tag_1, tag_2)] = -np.inf
            else:
                weights[(tag_1, tag_2)] = probs
    return weights

def compute_weights_variables(nb_weights, hmm_trans_weights, vocab, word_to_ix, tag_to_ix):
    """
    Computes autograd Variables of two weights: emission_probabilities and the tag_transition_probabilties
    parameters:
    nb_weights: -- a dictionary of emission weights
    hmm_trans_weights: -- dictionary of tag transition weights
    vocab: -- list of all the words
    word_to_ix: -- a dictionary that maps each word in the vocab to a unique index
    tag_to_ix: -- a dictionary that maps each tag (including the START_TAG and the END_TAG) to a unique index.
    
    :returns:
    emission_probs_vr: torch Variable of a matrix of size Vocab x Tagset_size
    tag_transition_probs_vr: torch Variable of a matrix of size Tagset_size x Tagset_size
    :rtype: autograd Variables of the the weights
    """
    # Assume that tag_to_ix includes both START_TAG and END_TAG
    tag_transition_probs = np.full((len(tag_to_ix), len(tag_to_ix)), -np.inf)
    emission_probs = np.full((len(vocab),len(tag_to_ix)), 0.0)

    for word in word_to_ix.keys():
        word__index = word_to_ix[word]
        for tag in tag_to_ix.keys():
            tag_index = tag_to_ix[tag]
            if (tag,word) in nb_weights.keys():
                emission_probs[word__index, tag_index] = nb_weights[(tag,word)]
            # elif tag != START_TAG and tag != END_TAG:
            #     emission_probs[word__index, tag_index] = 0
            elif tag == START_TAG or tag==END_TAG:
                emission_probs[word__index, tag_index] = -np.inf

    for tag1 in tag_to_ix.keys():
        tag_1_index = tag_to_ix[tag1]
        for tag2 in tag_to_ix.keys():
            tag_2_index = tag_to_ix[tag2]
            if (tag1, tag2) in hmm_trans_weights.keys():
                tag_transition_probs[tag_1_index, tag_2_index] = hmm_trans_weights[(tag1, tag2)]

    emission_probs_vr = Variable(torch.from_numpy(emission_probs.astype(np.float32)))
    tag_transition_probs_vr = Variable(torch.from_numpy(tag_transition_probs.astype(np.float32)))
    
    return emission_probs_vr, tag_transition_probs_vr
    
