from Utils import *
import string
import numpy as np

filename = 'deu.txt'
doc = load_doc(filename)
pairs = to_pairs(doc)
pairs = remove_garbage(pairs)
clean_pairs = clean_pairs(pairs)
save_clean_data(clean_pairs, 'english_german.pkl')
raw_dataset = load_clean_sentences('english_german.pkl')
n_sentences = 10000
dataset = raw_dataset[:n_sentences, :]
np.random.shuffle(dataset)
train, test = dataset[:9000], dataset[9000:]
save_clean_data(dataset, 'english_german-both.pkl')
save_clean_data(train, 'english_german-train.pkl')
save_clean_data(test, 'english_german-test.pkl')
