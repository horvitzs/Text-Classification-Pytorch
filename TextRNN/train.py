import os 
import sys
sys.path.append( os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data_utils import Dictionary, Corpus_20News, DataLoader_20News
import torch 

TRAIN_DATA_DIR = '../data/20news-bydate-train'
TEST_DATA_DIR = '../data/20news-bydate-test'

data_loader = DataLoader_20News()
corpus = Corpus_20News()
MAX_LENGTH = 1000 


if __name__=='__main__':

	texts, labels, labels_index = data_loader.load_data_labels(TRAIN_DATA_DIR)

	temp = corpus.get_data(texts[0], MAX_LENGTH)

	#a = corpus.dictionary
	#print(a.vectors[a.word2idx['word']])


	print(temp)

