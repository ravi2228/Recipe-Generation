import numpy as np
import torch
import torch.nn as nn
from ing_verb_ordering import constants, preproc, most_common, clf_base, evaluation
from ing_verb_ordering import scorer, tagger_base, naive_bayes, hmm, viterbi, bilstm

USE_CUDA = False
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
UNK = constants.UNK
# recalculating vocab: obtains the most common 6900 words from the file
vocab, word_to_ix = most_common.get_word_to_ix(TRAIN_FILE, 6900)
vocab_dev, word_to_ix_dev = most_common.get_word_to_ix(DEV_FILE, 6900)

for word in word_to_ix_dev.keys():
    if word not in word_to_ix.keys():
        word_to_ix[word] = len(word_to_ix)
all_words = set(vocab).union(set(vocab_dev))
vocab = list(all_words)
print('words in the vocabulary: ', len(word_to_ix))
print(word_to_ix[UNK])

if START_TAG in all_tags:
    all_tags.remove(START_TAG)
if END_TAG in all_tags:
    all_tags.remove(END_TAG)

tag_to_ix = {}
all_to_ix = {}
for tag in all_tags:
    if tag not in tag_to_ix:
        tag_to_ix[tag] = len(tag_to_ix)
    if tag not in all_to_ix:
        all_to_ix[tag] = len(all_to_ix)

for word in vocab:
    if word not in all_to_ix:
        all_to_ix[word] = len(all_to_ix)

X_tr, Y_tr = preproc.load_data_given_verbs(TRAIN_FILE)

#print(bilstm.prepare_sequence(X_tr[1], word_to_ix).data.numpy())
#print(bilstm.prepare_sequence(Y_tr[1], tag_to_ix).data.numpy())

# - Loading Dev data for english:

# In[18]:


X_dv, Y_dv = preproc.load_data_given_verbs(DEV_FILE)
# loading dev data


# - Loading Test data for english:

# In[19]:


X_te, Y_te = preproc.load_data_given_verbs(TEST_FILE)
# loading test data


# In[20]:


# dev_tags = preproc.get_all_tags(DEV_FILE)
# test_tags = preproc.get_all_tags(TEST_FILE)
# for tag in dev_tags:
#     if tag not in tag_to_ix:
#         tag_to_ix[tag] = len(tag_to_ix)
# for tag in test_tags:
#     if tag not in tag_to_ix:
#         tag_to_ix[tag] = len(tag_to_ix)

filename = 'data/polyglot-en.pkl'
word_embeddings = bilstm.obtain_polyglot_embeddings(filename, all_to_ix)


def cosine(emb1, emb2):  # function to return the cosine similarity between the embeddings
    return emb1.dot(emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


torch.manual_seed(765)
embedding_dim = 64
hidden_dim = 30
model = bilstm.BiLSTM(len(all_to_ix), all_to_ix, embedding_dim, hidden_dim, word_embeddings, USE_CUDA=False)
loss = torch.nn.CrossEntropyLoss()
model, losses, accuracies = bilstm.train_model(loss, model, X_tr, Y_tr, all_to_ix, all_to_ix,
                                               X_dv, Y_dv, num_its=20, status_frequency=1,
                                               optim_args={'lr': 0.001, 'momentum': 0}, param_file='best.params', USE_CUDA=False)
# bilstm.plot_results(losses, accuracies)

bilstm.plot_results(losses, accuracies, save_name='ordering_verbs_loss.png')

for i in range(5):
    sentence = bilstm.prepare_sequence(X_dv[i], all_to_ix, USE_CUDA=USE_CUDA)
    best_answer = model.predict(sentence)
    print('in: {}'.format(' '.join(X_dv[i])))
    print('out: {}'.format(' '.join(best_answer)))
    print('gt: {}'.format(' '.join(Y_dv[i])))
    print('\n')
