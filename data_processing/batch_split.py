import json
import ijson
from constants import (
    ING,
    ACT,
    ING_LIST_SMALL_PATH,
    RECIPE_PATH_SMALL,
    RECIPE_PATH,
    ING_LIST_PATH,
    BATCH_PATH
)

BATCH_SIZE = 10000
OUTFILE = '-ing_batch.json'

def split_into_batches(cookbook_file):
    current = []
    count = 0
    with open(cookbook_file, 'r') as json_file:
        for item in ijson.items(json_file, "item"):
            current.append(item)
            if count > 0 and count % BATCH_SIZE == 0:
                print("Finished Item " + str(count))
                with open(BATCH_PATH + str(int(count / BATCH_SIZE - 1)) + OUTFILE, 'w') as outfile:
                    json.dump(current, outfile)
                current = []
            count += 1


if __name__ == '__main__':
    #file = 'C:\\Users\super\Documents\\recipe_generation\data\det_ingrs.json'
    #nltk.download('averaged_perceptron_tagger')
    #c = OneMInfo(ING_LIST_PATH, RECIPE_PATH)
    #print(c.calculate_cooccurence_matrix(ING, ACT))
    #i = get_ingredient_list_from_file(file)
    #print(i[0:10])
    #c = OneMInfo(ING_LIST_PATH, RECIPE_PATH)
    split_into_batches(ING_LIST_PATH)
