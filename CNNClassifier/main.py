import os
import sys 
import torch
sys.path.append( os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pytorch_text.torchtext.data as data
import pytorch_text.torchtext.datasets as datasets
import pytorch_text.torchtext.vocab as vocab
import model 
import train 
from data_utils.MR import MR
from data_utils.News20 import News20
from nltk.corpus import stopwords

def SST_data_loader(text_field, label_field, vector, b_size, **kwargs):


	train_data, dev_data, test_data = datasets.SST.splits(text_field, label_field, fine_grained = True)
	text_field.build_vocab(train_data, dev_data, test_data, vectors= vector)
	label_field.build_vocab(train_data, dev_data, test_data, vectors = vector)

	# print information about the data
	print('len(train)', len(train_data))
	print('len(test)', len(test_data))



	train_loader, dev_loader, test_loader = data.BucketIterator.splits(
		(train_data, dev_data, test_data), batch_sizes = (b_size, len(dev_data), len(test_data)), **kwargs)



	return train_loader, dev_loader, test_loader

def MR_data_loader(text_field, label_field, vector, b_size, **kwargs):

	train_data, dev_data, test_data = MR.splits(text_field, label_field)
	text_field.build_vocab(train_data, dev_data, test_data, vectors= vector)
	label_field.build_vocab(train_data, dev_data, test_data, vectors = vector)

	# print information about the data
	print('len(train)', len(train_data))
	print('len(test)', len(test_data))

	train_loader, dev_loader, test_loader = data.BucketIterator.splits(
		(train_data, dev_data, test_data), batch_sizes = (b_size, len(dev_data), len(test_data)), **kwargs)



	return train_loader, dev_loader, test_loader

def News_20_data_loader(text_field, label_field, vector, b_size, **kwargs):

	train_data, dev_data, test_data = News20.splits(text_field, label_field)
	text_field.build_vocab(train_data, dev_data, test_data, vectors= vector)
	label_field.build_vocab(train_data, dev_data, test_data, vectors = vector)

	# print information about the data
	print('len(train)', len(train_data))
	print('len(test)', len(test_data))

	train_loader, dev_loader, test_loader = data.BucketIterator.splits(
		(train_data, dev_data, test_data), batch_sizes = (b_size, len(dev_data), len(test_data)), **kwargs)



	return train_loader, dev_loader, test_loader


def clean_str(strings):
    stop_words = list(set(stopwords.words('english')))
    stop_characters = ["`", "\'", "\"", ".", "\(", "\)", "," , '``', "''", '--', '...']
    stop_words.extend(stop_characters)
    filtered_words = [word for word in strings if word not in stop_words]
    return filtered_words


if __name__=='__main__':

	   
    #glove 6B 100 dim / glove 6B 300 dim /glove 42B 300 dim 
	glove = vocab.GloVe(name = '6B', dim = 100)
	iscuda = True
	device_value = -1  	#device = - 1 : cpu 
	batch_size = 20


	if torch.cuda.is_available() is True:
		iscuda = True
	else:
		iscuda = False
		#device_value = -1 



	#load data
	print("Load data...")
	# to fix length : fix_length = a 
	text_field = data.Field(lower = True, batch_first = True, fix_length = 200, preprocessing = clean_str)
	label_field = data.Field(sequential = False)

    #select data set 
	train_loader, dev_loader, test_loader = News_20_data_loader(text_field, label_field, glove, batch_size, device = device_value, repeat = False)
	#train_loader, dev_loader, test_loader = SST_data_loader(text_field, label_field, glove, batch_size, device = device_value, repeat = False)
	#train_loader, dev_loader, test_loader = MR_data_loader(text_field, label_field, glove, batch_size, device = device_value, repeat = False)


    #parameters 
	in_channels = 1 
	out_channels = 2
	voca_size = len(text_field.vocab)
	num_classes = len(label_field.vocab) - 1 
	embed_dim = glove.vectors.size()[1]
	kernel_sizes = [3,4,5]
	dropout_p = 0.7
	embedding_weight = text_field.vocab.vectors

	learnign_rate = 0.001
	num_epochs = 100

	#parameter of rnn 
	num_layer  = 2 
	num_hidden = 128


	# model 
	print("Load model...")
	#classifier_model = model.CNNClassifier(in_channels, out_channels, voca_size, embed_dim, num_classes, kernel_sizes, dropout_p, embedding_weight)
	classifier_model = model.RNNClassifier(voca_size, embed_dim, num_hidden, num_layer, num_classes, embedding_weight)
	if iscuda:
		classifier_model = classifier_model.cuda()

	# train 
	print("Start Train...")
	train.train(train_loader, dev_loader, classifier_model, iscuda, learnign_rate, num_epochs)

	# eval 
	print("Evaluation")
	train.eval(test_loader, classifier_model, iscuda) 


	