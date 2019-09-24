# If you change the location of dataset, then
# Make the appropriate changes to the following constants.
CWD = '/home/twister/Documents/caras_xiao/RecipeGeneration'
TAGGER_DATA_PATH = CWD + '/data_processing/data/tagger_data'
TRAIN_FILE_TEMPLATE = TAGGER_DATA_PATH + '/{}_ing_nt_verb_train.txt'
TEST_FILE_TEMPLATE = TAGGER_DATA_PATH + '/{}_ing_nt_verb_test.txt'
DEV_FILE_TEMPLATE = TAGGER_DATA_PATH + '/{}_ing_nt_verb_valid.txt'
TRAIN_FILE_TEMPLATE_TITLE = TAGGER_DATA_PATH + '/{}_ing_all_verb_train.txt'
TEST_FILE_TEMPLATE_TITLE = TAGGER_DATA_PATH + '/{}_ing_all_verb_test.txt'
DEV_FILE_TEMPLATE_TITLE = TAGGER_DATA_PATH + '/{}_ing_all_verb_valid.txt'
TRAIN_FILE = TAGGER_DATA_PATH + '/31_vs_ing_verb_train.txt'
DEV_FILE = TAGGER_DATA_PATH + '/31_vs_ing_verb_valid.txt'
#TEST_FILE_UNLABELED = 'data/en-ud-test-hidden.conllu'
TEST_FILE = TAGGER_DATA_PATH + '/31_vs_ing_verb_test.txt'
TITLE = 'title'
OFFSET = '**OFFSET**'
start_batch = 31
num_batches = 14
START_TAG = '<start>'
END_TAG = '<end>'


UNK = '<UNK>' 
