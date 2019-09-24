import operator
from collections import defaultdict, Counter
from ing_verb_ordering.constants import START_TAG,END_TAG, UNK
import numpy as np
import torch
import torch.nn
import math
from torch import autograd
from torch.autograd import Variable

def get_torch_variable(arr):
    # returns a pytorch variable of the array
    torch_var = torch.autograd.Variable(torch.from_numpy(np.array(arr).astype(np.float32)))
    return torch_var.view(1,-1)

def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def viterbi_step(all_tags, tag_to_ix, cur_tag_scores, transition_scores, prev_scores):
    """
    Calculates the best path score and corresponding back pointer for each tag for a word in the sentence in pytorch, which you will call from the main viterbi routine.
    
    parameters:
    - all_tags: list of all tags: includes both the START_TAG and END_TAG
    - tag_to_ix: a dictionary that maps each tag (including the START_TAG and the END_TAG) to a unique index.
    - cur_tag_scores: pytorch Variable that contains the local emission score for each tag for the current token in the sentence
                       it's size is : [ len(all_tags) ] 
    - transition_scores: pytorch Variable that contains the tag_transition_scores
                        it's size is : [ len(all_tags) x len(all_tags) ] 
    - prev_scores: pytorch Variable that contains the scores for each tag for the previous token in the sentence: 
                    it's size is : [ 1 x len(all_tags) ] 
    
    :returns:
    - viterbivars: a list of pytorch Variables such that each element contains the score for each tag in all_tags for the current token in the sentence
    - bptrs: a list of idx that contains the best_previous_tag for each tag in all_tags for the current token in the sentence

    size of viterbivars: [ len(all_tags) x 1]: each indicating the best score resulting for that particular tag for the current token.
    size of bptrs: [ len(all_tags) x 1]: each indicating the previous tag that resulted in the best score for each tag for the current token.
    """

    bptrs = []
    viterbivars=[]
    for next_tag in list(all_tags):
        best_so_far_index = 0
        best_so_far = None
        first = True
        next_tag_index = tag_to_ix[next_tag]
        for cur_tag in list(all_tags):
            cur_tag_index = tag_to_ix[cur_tag]
            instance_best = prev_scores[0, cur_tag_index] + transition_scores[next_tag_index, cur_tag_index] + cur_tag_scores[next_tag_index]
            if first or instance_best.item() > best_so_far.item():
                first = False
                best_so_far = instance_best
                best_so_far_index = cur_tag_index
        viterbivars.append(best_so_far)
        bptrs.append(best_so_far_index)
    return viterbivars, bptrs

def build_trellis(all_tags, tag_to_ix, cur_tag_scores, transition_scores):
    """
    This function should compute the best_path and the path_score. 
    Use viterbi_step to implement build_trellis in viterbi.py in Pytorch.
    
    parameters:
    - all_tags: a list of all tags: includes START_TAG and END_TAG
    - tag_to_ix: a dictionary that maps each tag to a unique id.
    - cur_tag_scores: a list of pytorch Variables where each contains the local emission score for each tag for that particular token in the sentence, len(cur_tag_scores) will be equal to len(words)
                        it's size is : [ len(words in sequence) x len(all_tags) ] 
    - transition_scores: pytorch Variable (a matrix) that contains the tag_transition_scores
                        it's size is : [ len(all_tags) x len(all_tags) ] 
    
    :returns:
    - path_score: the score for the best_path
    - best_path: the actual best_path, which is the list of tags for each token: exclude the START_TAG and END_TAG here.
    """
    
    ix_to_tag={ v:k for k,v in tag_to_ix.items() }
    
    # setting all the initial score to START_TAG
    # make sure END_TAG is in all_tags
    initial_vec = np.full((1,len(all_tags)),-np.inf)
    initial_vec[0][tag_to_ix[START_TAG]] = 0
    prev_scores = torch.autograd.Variable(torch.from_numpy(initial_vec.astype(np.float32))).view(1,-1)
    whole_bptrs=[]
    whole_viter = []
    viterbivars = []
    for m in range(len(cur_tag_scores)):
        viterbivars, bptrs = viterbi_step(all_tags, tag_to_ix, cur_tag_scores[m], transition_scores, prev_scores)
        prev_scores = get_torch_variable(viterbivars)
        whole_viter.append(viterbivars)
        whole_bptrs.append(bptrs)
    # after you finish calculating the tags for all the words: don't forget to calculate the scores for the END_TAG
    end_index = tag_to_ix[END_TAG]
    path_score = 0
    first = True
    came_from = -1
    for tag in list(all_tags):
        tag_index = tag_to_ix[tag]
        a = viterbivars[tag_index] + transition_scores[end_index,tag_index]
        if first or a.item() > path_score.item():
            first = False
            path_score = a
            came_from = tag_index

    # Calculate the best_score and also the best_path using backpointers and don't forget to reverse the path
    best_path = len(whole_bptrs)*[""]
    for i in range(len(whole_bptrs)):
        best_path[len(whole_bptrs)-i-1] = ix_to_tag[came_from]
        pt = whole_bptrs[len(whole_bptrs)-i-1]
        came_from = pt[came_from]
    return path_score, best_path

    
