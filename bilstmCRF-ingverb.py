from ing_verb_ordering import constants, preproc, most_common, clf_base, evaluation
from ing_verb_ordering import scorer, tagger_base, naive_bayes, hmm, viterbi, bilstm
import torch

TRAIN_FILE = constants.TRAIN_FILE
DEV_FILE = constants.DEV_FILE
TEST_FILE = constants.TEST_FILE
START_TAG = constants.START_TAG
END_TAG = constants.END_TAG
TITLE = constants.TITLE

all_tags = preproc.get_all_tags(TRAIN_FILE)
all_tags_dev = preproc.get_all_tags(DEV_FILE)
all_tags_tst = preproc.get_all_tags(TEST_FILE)

all_tags = all_tags.union(all_tags_dev)
all_tags = all_tags.union(all_tags_tst)
all_tags.add(TITLE)
UNK = constants.UNK
# recalculating vocab: obtains the most common 6900 words from the file
vocab, word_to_ix = most_common.get_word_to_ix(TRAIN_FILE, 6900)
print('words in the vocabulary: ', len(word_to_ix))
print(word_to_ix[UNK])

USE_CUDA = False

if START_TAG in all_tags:
    all_tags.remove(START_TAG)
if END_TAG in all_tags:
    all_tags.remove(END_TAG)
all_tags = sorted(all_tags)

tag_to_ix = {}
for tag in all_tags:
    if tag not in tag_to_ix:
        tag_to_ix[tag] = len(tag_to_ix)

X_tr, Y_tr = preproc.load_data(TRAIN_FILE)
X_dv, Y_dv = preproc.load_data(DEV_FILE)
#X_te, Y_te = preproc.load_data(TEST_FILE_HIDDEN)


if START_TAG not in tag_to_ix:
    tag_to_ix[START_TAG] = len(tag_to_ix)
if END_TAG not in tag_to_ix:
    tag_to_ix[END_TAG] = len(tag_to_ix)

embedding_dim= 64
hidden_dim= 2
filename = 'data/polyglot-en.pkl'
word_embeddings = bilstm.obtain_polyglot_embeddings(filename, word_to_ix)
model = bilstm.BiLSTM_CRF(len(word_to_ix),tag_to_ix,embedding_dim, hidden_dim, embeddings=word_embeddings)
print (model)
if USE_CUDA:
    model.cuda()
loss = model.neg_log_likelihood
model, losses, accuracies = bilstm.train_model(loss, model, X_tr,Y_tr, word_to_ix, tag_to_ix,
                                               X_dv, Y_dv,
                                               num_its=1, status_frequency=1,
                                               optim_args = {'lr':0.1,'momentum':0}, param_file = 'best.params', USE_CUDA=USE_CUDA)
bilstm.plot_results(losses, accuracies, save_name='bilstm_crf.png')
torch.save(model.state_dict(), 'bilstm_crf.pt')
