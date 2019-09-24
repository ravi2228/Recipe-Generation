
# coding: utf-8

# In[ ]:


import os
import nltk
import pandas as pd
import numpy as np
import json
import ijson
import numpy as np
import gensim.models.keyedvectors as word2vec

from skimage import io, transform



#csv file containing image id's
kyle_data = pd.read_csv('all_kyle_image_ids.csv')


image_id = kyle_data['image'].tolist()



id_convert_dict = {}
with open('layer2.json', 'r') as json_file:
    for item in ijson.items(json_file, "item"):
        for check in item['images']:
            if check['id'] in image_id:  ## image id 
                print(item['id'])
                id_convert_dict[check['id']] = item['id'] ## layer 2


print('completed')



all_ids_layer2 = list(id_convert_dict.values())

landmarks_frame = pd.DataFrame(columns=['id','ingredients', 'image'])
curr_row = 0
with open('layer1.json', 'r') as json_file:
    for item in ijson.items(json_file, "item"):
        if item['id'] in all_ids_layer2: ## for the recipe id is contained all_ids_layers2
            landmarks_frame.loc[curr_row] = [item['id'],",".join([instr['text'] for instr in item['ingredients']]), [k for k,v in id_convert_dict.items() if v == item['id']][0]]
            curr_row += 1
            print(curr_row)


stopwords = ['tsp',"cook","cooked","dry","dried",'canned',"Diced","Dice","stock","whole","fillets","slice","sliced","pieces","frozen","and","g","/2","/4","PHILADELPHIA","white","-serving","size","serving","serving size","-serving size","taste","less","more","fresh","can","packed","package","pack","I","ml","for","the","is","used","mix","tablespoons","tablespoon",'lb','lbs','of','oz','-oz','pkg','tbsp',"package","large","small","medium","grams","half","halves","quarter","quartered","small","cut","into","to","add","1/2", "3/4","1/4", "tbsp.", "cup", "cups", "/", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0","c.","10", "teaspoon","or", "(", ")","teaspoons", "ounce.", "ounces", "ounce", "box"]


landmarks_frame['new_ing'] = landmarks_frame['ingredients'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stopwords]))

curr_row=0
f = []
for i in range(len(landmarks_frame['new_ing'])):
    d=[]
    for word in landmarks_frame['new_ing'][i].split(','):
        d.append(" ".join([w for w in word.split() if w.isalpha()]))
    f.append(",".join([w for w in d]))
    #curr_row += 1
    

df2 = pd.DataFrame(f)
df2.columns = ['ing']


merged = pd.merge(left=landmarks_frame, left_index=True,
                  right=df2, right_index=True,
                  how='inner')


del merged['ingredients']
del merged['new_ing']


merged.to_pickle("./train_ingredients.pkl")

