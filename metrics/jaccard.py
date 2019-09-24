import pandas as pd
from nltk.translate.bleu_score import sentence_bleu

data_df = pd.read_csv('output_ingredients_all_nofiller_google.txt')
data_df.dropna(inplace=True)

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def union(lst1, lst2):
    #final_list = list(set(lst1) | set(lst2))
    #return final_list
    final_list = lst1
    for item in lst2:
        if item not in final_list:
            final_list.append(item)
    return final_list

class word:
    def __init__(self, x):
        self.val = x
    def __eq__(self, other):
        if self.val in other.val:
            return True
        if other.val in self.val:
            return True
        return False

scores = []
for index, row in data_df.iterrows():
    ing_list = [word(x.strip()) for x in row['ing'].split(',')]
    prediction_list = [word(x.strip()) for x in row['prediction'].split(' ') if x != '<start>' and x!= '<end>']
    ing_list.sort()
    prediction_list.sort()
    current = 1.0 * len(intersection(prediction_list, ing_list)) / len(union(prediction_list, ing_list))
    scores.append(current)

print(sum(scores) / len(scores))
