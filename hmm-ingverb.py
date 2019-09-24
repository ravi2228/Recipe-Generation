
import sys
from importlib import reload


# In[3]:


print('My Python version')

print('python: {}'.format(sys.version))


# In[4]:


import nose

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim


# importing all the necessary files from gtnlplib
from ing_verb_ordering import constants, preproc, most_common, clf_base, evaluation
from ing_verb_ordering import scorer, tagger_base, naive_bayes, hmm, viterbi, bilstm



## Define the file names
TRAIN_FILE = constants.TRAIN_FILE
DEV_FILE = constants.DEV_FILE
TEST_FILE = constants.TEST_FILE
# change the constant to test_file_unlabeled for release


# - Here is demo code for using the function `conll_seq_generator(...)`.
# - The default value for max_insts is `1000000` indicating the num. of instances: and this should be enough for our dataset.

# In[8]:


# ## Demo
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


# In[17]:


tagger_mc = tagger_base.make_classifier_tagger(theta_mc)
