import numpy as np
import torch
import torch.nn as nn
from ing_verb_ordering import constants, preproc, most_common, clf_base, evaluation
from ing_verb_ordering import scorer, tagger_base, naive_bayes, hmm, viterbi, bilstm
import csv

TRAIN_FILE = constants.TRAIN_FILE
DEV_FILE = constants.DEV_FILE
TEST_FILE = constants.TEST_FILE
TRAIN_FILE_TEMPLATE = constants.TRAIN_FILE_TEMPLATE
DEV_FILE_TEMPLATE = constants.DEV_FILE_TEMPLATE
TEST_FILE_TEMPLATE = constants.TEST_FILE_TEMPLATE
start_batch_num = constants.start_batch
num_batches = constants.num_batches

START_TAG = constants.START_TAG
END_TAG = constants.END_TAG
TITLE = constants.TITLE
all_tags = set()

for i in range(start_batch_num, start_batch_num + num_batches):
    all_tags_train = preproc.get_all_tags(TRAIN_FILE_TEMPLATE.format(i))
    all_tags_dev = preproc.get_all_tags(DEV_FILE_TEMPLATE.format(i))
    all_tags_tst = preproc.get_all_tags(TEST_FILE_TEMPLATE.format(i))
    all_tags = all_tags.union(all_tags_dev)
    all_tags = all_tags.union(all_tags_tst)
    all_tags = all_tags.union(all_tags_train)

UNK = constants.UNK
# recalculating vocab: obtains the most common 6900 words from the file
# word_to_ix = {}
# vocab = set()
# for i in range(start_batch_num, start_batch_num+num_batches):
#     vocab_train, word_to_ix_train = most_common.get_word_to_ix(TRAIN_FILE_TEMPLATE.format(i), 6900)
#     vocab_dev, word_to_ix_dev = most_common.get_word_to_ix(DEV_FILE_TEMPLATE.format(i), 6900)
#     vocab = vocab.union(vocab_train)
#     vocab = vocab.union(vocab_dev)
#
# for word in vocab:
#     word_to_ix[word] = len(word_to_ix)
word_to_ix, vocab, tag_to_ix, all_tags, X_tr, Y_tr, X_dv, Y_dv, X_te, Y_te = preproc.load_input_multiple_batch(start_batch_num, num_batches, include_title=True)
print('words in the vocabulary: ', len(word_to_ix))
print(word_to_ix[UNK])
USE_CUDA = False
all_tags.add(START_TAG)
all_tags.add(END_TAG)
# if START_TAG in all_tags:
#     all_tags.remove(START_TAG)
# if END_TAG in all_tags:
#     all_tags.remove(END_TAG)
# all_tags.add(TITLE)



# X_tr = []
# X_dv = []
# X_te = []
# Y_tr = []
# Y_dv = []
# Y_te = []
#
# for i in range(start_batch_num, start_batch_num+num_batches):
#     X_tr_i, Y_tr_i = preproc.load_data_no_title(TRAIN_FILE_TEMPLATE.format(i))
#     X_dv_i, Y_dv_i = preproc.load_data_no_title(DEV_FILE_TEMPLATE.format(i))
#     X_te_i, Y_te_i = preproc.load_data_no_title(TEST_FILE_TEMPLATE.format(i))
#     X_tr.extend(X_tr_i)
#     X_dv.extend(X_dv_i)
#     X_te.extend(X_te_i)
#     Y_tr.extend(Y_tr_i)
#     Y_dv.extend(Y_dv_i)
#     Y_te.extend(Y_te_i)


filename = 'data/polyglot-en.pkl'
word_embeddings = bilstm.obtain_polyglot_embeddings(filename, word_to_ix)


def cosine(emb1, emb2):  # function to return the cosine similarity between the embeddings
    return emb1.dot(emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


torch.manual_seed(765)
embedding_dim = 64
hidden_dim = 30
model = bilstm.BiLSTM(len(word_to_ix), tag_to_ix, embedding_dim, hidden_dim, word_embeddings, USE_CUDA=USE_CUDA)
loss = torch.nn.CrossEntropyLoss()
#model.load_state_dict(torch.load('bilstm_small_for_poster_title.pt'))
model, losses, accuracies, accuracies_train = bilstm.train_model(loss, model, X_tr, Y_tr, word_to_ix, tag_to_ix,
                                               X_dv, Y_dv, num_its=5, status_frequency=1,
                                               optim_args={'lr': 0.0001, 'momentum': 0}, param_file='best.params', USE_CUDA=False)

torch.save(model.state_dict(), 'bilstm_small_for_poster_title.pt')
bilstm.plot_results(losses, accuracies, save_name='bilstm_title_all_batches.png')
bilstm.plot_results_with_train_acc(losses, train_accuracies=accuracies_train, dev_accuracies=accuracies, save_name='bilstm_title_train_dev_plot.png')
#bilstm.plot_results(losses, accuracies, save_name='ordering_verbs_loss_title.png')
#for_the_gold = '<start> spaghetti spaghetti sauce garlic clove minced garlic minced dried oregano salt pepper <end>'
for_the_fold2 = '<start> Cheesy Creamy Potatoes potatoes onion salt cheddar cheese <end>'
for_the_gold3 = '<start> Mussels in Saffron mussels mussels mussels onion parsley thyme thyme bay leaf bay leaf olive oil mussels <end>'
goldy = [for_the_fold2, for_the_gold3]
for gold in goldy:
    sentence = bilstm.prepare_sequence(gold.split(' '), word_to_ix, USE_CUDA=USE_CUDA)
    best_answer = model.predict(sentence)
    print('in: {}'.format(gold))
    print('out: {}'.format(' '.join(best_answer)))
for i in range(6,10):
    sentence = bilstm.prepare_sequence(X_dv[i], word_to_ix, USE_CUDA=USE_CUDA)
    best_answer = model.predict(sentence)
    print('in: {}'.format(' '.join(X_dv[i])))
    print('out: {}'.format(' '.join(best_answer)))
    print('gt: {}'.format(' '.join(Y_dv[i])))
    print('\n')


# print('test sentences!!!')
# tst1 = 'potatoes butter olive oil salt pepper'
# tst2 = 'wheat baking powder baking soda salt ground cinnamon ground nutmeg ground cloves ground nutmeg butter softened eggs vanilla extract'
# tst3 = 'butter softened brown eggs vanilla extract baking soda baking soda salt chocolate chips'
# tst4 = 'banana peeled banana ice cream'
# tst5 = 'apple peeled cored sliced apple peeled cored chopped apple cider vinegar lemon juice'
# tst6 = 'wine,tarragon,rosemary,thyme,vegetable oil,garlic,minced,chicken'

# tsts = [tst1, tst2, tst3, tst4, tst5, tst6]
# print('\n')
# for i in tsts:
#     seq = i.split(' ')
#     # seq = ' '.split(row['prediction'])
#     # seq = '--START-- ' + row['ing'] + '--END--'
#     sentence = bilstm.prepare_sequence(seq, word_to_ix, USE_CUDA=USE_CUDA)
#     best_answer = model.predict(sentence)
#     print('in: {}'.format(' '.join(seq)))
#     print('out: {}'.format(' '.join(best_answer)))
#     #print('gt: {}'.format(' '.join(Y_dv[i])))
#     print('\n')
# out_file = 'output_ingredient_verb_pairs.csv'
# with open('output_ingredients.csv') as csv_file:
#     with open(out_file, mode = 'w') as of:
#
#         csv_reader = csv.DictReader(csv_file, delimiter=',')
#         line_count = 0
#         fieldnames = csv_reader.fieldnames
#         fieldnames.append('verbs')
#         csv_writer = csv.DictWriter(of, fieldnames=fieldnames)
#         for row in csv_reader:
#             if line_count == 0:
#                 csv_writer.writeheader()
#             else:
#                 seq = row['prediction'].split(' ')
#                 #seq = ' '.split(row['prediction'])
#                 #seq = '--START-- ' + row['ing'] + '--END--'
#                 sentence = bilstm.prepare_sequence(seq, word_to_ix, USE_CUDA=USE_CUDA)
#                 best_answer = model.predict(sentence)
#                 out_dict = {
#                     '':row[''],
#                     'id':row['id'],
#                     'image':row['image'],
#                     'ing':row['ing'],
#                     'prediction':row['prediction'],
#                     'verbs': ','.join(best_answer)
#                 }
#                 csv_writer.writerow(out_dict)
#                 print('in: {}'.format(' '.join(seq)))
#                 print('out: {}'.format(' '.join(best_answer)))
#                 # print('gt: {}'.format(' '.join(Y_dv[i])))
#                 print('\n')
#             line_count += 1

# for i in range(len(tsts)):
#     seq = '--START-- '+ tsts[i] + ' --END--'
#     sentence = bilstm.prepare_sequence(seq.split(' '), word_to_ix, USE_CUDA=USE_CUDA)
#     best_answer = model.predict(sentence)
#     print('in: {}'.format('--START-- '+ tsts[i] + ' --END--'))
#     print('out: {}'.format(' '.join(best_answer)))
#     #print('gt: {}'.format(' '.join(Y_dv[i])))
#     print('\n')