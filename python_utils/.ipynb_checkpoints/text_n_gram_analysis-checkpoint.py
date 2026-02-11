import os
import re
import json
import numpy as np
import pandas as pd 
from tqdm import tqdm
import matplotlib.pyplot as plt

data_path = '/data/gusev/USERS/jpconnor/clinical_text_project/data/'

IO_notes_data = json.load(open(data_path + 'IO_notes_with_full_metadata.json', 'r'))
text_data = IO_notes_data['CLINICAL_TEXT']

n = 30
n_gram_dict = dict()
for text_entry in tqdm(text_data):
    word_list = re.split('\s+', re.sub(r'\W+', ' ', text_entry).lower())

    for i in range(len(word_list) - n):
        cur_n_gram = ' '.join(word_list[i:i+n])

        if cur_n_gram in n_gram_dict.keys():
            n_gram_dict[cur_n_gram] += 1
        else:
            n_gram_dict[cur_n_gram] = 1
            
sorted_dict = {key: value for key, value in sorted(n_gram_dict.items(), key=lambda item: item[1], reverse=True)}

dumped_dict = json.dumps(sorted_dict)
_ = open(data_path + f'clinical_text_{n}_gram_occurrences.json', 'w').write(dumped_dict)

sorted_10_gram_dict = json.load(open(data_path + 'clinical_text_10_gram_occurrences.json', 'r'))
results_above_threshold = [val for val in sorted_10_gram_dict.values() if val > 25]

plt.figure(figsize=(10,8))
plt.plot(range(len(results_above_threshold)), results_above_threshold)
plt.title('10-gram frequency')
plt.xlabel('hit rank')
plt.ylabel('num occurrences per hit')