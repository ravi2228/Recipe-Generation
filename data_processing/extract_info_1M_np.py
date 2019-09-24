import json
from constants import (
    ING,
    ACT,
    ING_LIST_SMALL_PATH,
    RECIPE_PATH_SMALL,
    RECIPE_PATH,
    ING_BATCH_PATH,
    RECIPE_BATCH_PATH,
    ING_LIST_PATH
)
import nltk
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer


class OneMInfo:
    def __init__(self, ing_file, cookbook_file):
        self.ing_file = ing_file
        self.recipe_id_to_ix = {}
        self.ingredient_to_ix = {}
        self.verb_to_ix = {}
        self.ing_verb_pairs = []
        self.recipe_coded_ingredients = None
        self.recipe_coded_verbs = None
        self.recipe_ingredient_list = {}
        self.verb_exclude_list = ['is', 'be', 'use', 'do']
        self.cookbook_file = cookbook_file
        self.ingredient_list = self.extract_ingredients()
        self.verb_list = self.extract_verbs()
        self.cookbook = self.get_recipes()

    def extract_verbs(self):
        verb_list = []
        i = 0
        with open(self.cookbook_file, 'r') as json_file:
            data = json.load(json_file)
            for recipe in data:
                for instr in recipe['instructions']:
                    sents = nltk.sent_tokenize(instr['text'])
                    for sent in sents:
                        word_pos = nltk.pos_tag(nltk.word_tokenize(sent))
                        verbs_in_instr = [WordNetLemmatizer().lemmatize(word,'v').lower().strip() for word, pos in word_pos if 'V' in pos if WordNetLemmatizer().lemmatize(word,'v') not in self.verb_exclude_list]
                        verb_list.extend(verbs_in_instr)
        verb_list = set(verb_list)
        for verb in verb_list:
            self.verb_to_ix[verb] = i
            i += 1
        return verb_list

    def extract_ingredients(self):
        all_ing = []
        i=0
        print("extracting ingredients")
        with open(self.ing_file, 'r') as json_file:
            data = json.load(json_file)
            print("loaded ingredient file")
            for recipe in data:
                self.recipe_ingredient_list[recipe['id']] = recipe[ING]
                for ingredient in recipe[ING]:
                    all_ing.append(ingredient["text"].lower().strip())
        all_ing = set(all_ing)
        for ing in all_ing:
            self.ingredient_to_ix[ing] = i
            i += 1
        return set(all_ing)

    def get_recipes(self):
        cookbook = {}
        i = 0
        with open(self.cookbook_file, 'r') as json_file:
            data = json.load(json_file)
            recipe_coded_ingredients = np.zeros((len(data), len(self.ingredient_list)))
            recipe_coded_verbs = np.zeros((len(data), len(self.verb_list)))
            for recipe in data:
                cookbook[recipe['id']] = {}
                cookbook["row_ix"] = i
                #ing_verb = [recipe['id']+',']
                ing_verb = [recipe['title']+',']
                appeared = []
                for recipe_key in recipe.keys():
                    if recipe_key == ING:
                        ix_list = [self.ingredient_to_ix[ingredient['text'].lower().strip()] for ingredient in self.recipe_ingredient_list[recipe['id']]]
                        recipe_coded_ingredients[i, ix_list] = 1
                        #ix_list = [self.verb_to_ix[verb]]
                        #word_pos = pos_tag(word_tokenize(instr['text']))
                    if recipe_key == 'instructions':
                        if len(recipe['instructions']) > 0:
                            for instr in recipe['instructions']:
                                sents = nltk.sent_tokenize(instr['text'])
                                for sent in sents:
                                    sent_tokened = nltk.word_tokenize(sent)
                                    word_pos = nltk.pos_tag(sent_tokened)
                                #word_pos = nltk.pos_tag(nltk.word_tokenize(instr['text']))
                                    verb_list = [WordNetLemmatizer().lemmatize(word,'v').lower().strip() for word, pos in word_pos if 'V' in pos and WordNetLemmatizer().lemmatize(word,'v') not in self.verb_exclude_list]
                                    if len(verb_list) == 0:
                                        verb_list = [WordNetLemmatizer().lemmatize(nltk.pos_tag([word])[0][0], 'v').lower().strip() for
                                                       word in sent_tokened if 'V' in nltk.pos_tag([word])[0][1]
                                                       ]
                                    else:
                                        verb_list_oc = [verb for verb in verb_list if 'V' in nltk.pos_tag([verb])[0][1]]
                                        if len(verb_list_oc) > 0:
                                            verb_list = verb_list_oc
                                    ix_list = []
                                    for verb in verb_list:
                                        if verb not in self.verb_to_ix.keys():
                                            self.verb_to_ix[verb] = len(self.verb_to_ix.keys())
                                            self.verb_list.add(verb)
                                            new_col = np.zeros((len(data), 1))
                                            recipe_coded_verbs = np.hstack((recipe_coded_verbs, new_col))
                                        ix_list.append(self.verb_to_ix[verb])
                                    #ix_list = [self.verb_to_ix[verb] for verb in verb_list]
                                    for ingredient in self.recipe_ingredient_list[recipe['id']]:
                                        if ingredient['text'] in sent:
                                            ing_verb.extend(['{} :: {}, '.format(ingredient['text'].lower().strip(), WordNetLemmatizer().lemmatize(verb,'v').lower().strip()) for verb in verb_list if WordNetLemmatizer().lemmatize(verb,'v') not in self.verb_exclude_list and WordNetLemmatizer().lemmatize(verb,'v').lower().strip() not in self.ingredient_list])
                                            appeared.append(ingredient['text'])
                                    recipe_coded_verbs[i, ix_list] = 1
                            cookbook[recipe['id']][recipe_key] = recipe[recipe_key]
                        else:
                            print('here')
                    else:
                        cookbook[recipe['id']][recipe_key] = recipe[recipe_key]
                if len(ing_verb) > 1:# and len(appeared) == len(self.recipe_ingredient_list[recipe['id']]):
                    self.ing_verb_pairs.append(ing_verb)
                # if len(appeared) < len(self.recipe_ingredient_list[recipe['id']]):
                #     print('here')
                i += 1
        self.recipe_coded_ingredients = recipe_coded_ingredients
        self.recipe_coded_verbs = recipe_coded_verbs
        return cookbook

    def print_ing_verb_pairs(self, train_per, batch_num):
        train_out_file = 'data\\tagger_data\{}_ing_all_verb_train.txt'.format(batch_num)
        valid_out_file = 'data\\tagger_data\{}_ing_all_verb_valid.txt'.format(batch_num)
        test_out_file = 'data\\tagger_data\{}_ing_all_verb_test.txt'.format(batch_num)

        train_lines = [' '.join(pairing_list) for pairing_list in self.ing_verb_pairs[:int(len(self.ing_verb_pairs)*train_per*0.7)]]
        valid_lines = [' '.join(pairing_list) for pairing_list in self.ing_verb_pairs[int(len(self.ing_verb_pairs)*train_per*0.7):int(len(self.ing_verb_pairs)*train_per*0.7) + int(len(self.ing_verb_pairs)*train_per*0.3)]]
        test_lines = [' '.join(pairing_list) for pairing_list in self.ing_verb_pairs[int(len(self.ing_verb_pairs)*train_per):]]

        train_file = open(train_out_file, 'w')
        train_file.write('\n'.join(train_lines))
        train_file.close()
        test_file = open(test_out_file, 'w')
        test_file.write('\n'.join(test_lines))
        test_file.close()
        valid_file = open(valid_out_file, 'w')
        valid_file.write('\n'.join(valid_lines))
        valid_file.close()

    def calculate_cooccurence_matrix(self, thing1, thing2):
        if thing1 == ING and thing2 == ING:
            listy = self.ingredient_list
            recipe_coded = self.recipe_coded_ingredients
        else:
            listy = self.verb_list
            recipe_coded = self.recipe_coded_verbs

        ing_matrix = np.zeros((len(listy), len(listy)))
        for ing1_ix in range(len(listy)):
            ig1_col = recipe_coded[:, ing1_ix]
            for ing2_ix in range(len(listy)):
                ig2_col = recipe_coded[:, ing2_ix]
                ing_matrix[ing1_ix, ing2_ix] = sum(np.logical_and(ig1_col, ig2_col))
        return ing_matrix



if __name__ == '__main__':
    #file = 'C:\\Users\super\Documents\\recipe_generation\data\det_ingrs.json'
    #nltk.download('averaged_perceptron_tagger')
    #print(nltk.pos_tag(['macaroni']))
    for i in range(31, 48):
        print("working on batch: {}".format(i))
        ing_path = ING_BATCH_PATH + '\\recipe_batches{}-ing_batch.json'.format(i)
        rec_path = RECIPE_BATCH_PATH + '\\{}-batch.json'.format(i)
        c = OneMInfo(ing_path, rec_path)
        c.print_ing_verb_pairs(0.8, i)
    #print(c.calculate_cooccurence_matrix(ING, ACT))
    #i = get_ingredient_list_from_file(file)
    #print(i[0:10])
