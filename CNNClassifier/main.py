import os
import sys 
import torch
sys.path.append( os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pytorch_text.torchtext.data as data
import pytorch_text.torchtext.datasets as datasets
import pytorch_text.torchtext.vocab as vocab
import model 
import train 

def SST_data_loader(text_field, label_field, vector, b_size, **kwargs):


	train_data, dev_data, test_data = datasets.SST.splits(text_field, label_field, fine_grained = True)
	text_field.build_vocab(train_data, dev_data, test_data, vectors= vector)
	label_field.build_vocab(train_data, dev_data, test_data, vectors = vector)

	train_loader, dev_loader, test_loader = data.BucketIterator.splits(
		(train_data, dev_data, test_data), batch_sizes = (b_size, len(dev_data), len(test_data)), **kwargs)



	return train_loader, dev_loader, test_loader




if __name__=='__main__':


	glove = vocab.GloVe(name='6B', dim=100)
	iscuda = True
	device_value = -1
	batch_size = 20


	if torch.cuda.is_available() is True:
		iscuda = True
	else:
		iscuda = False
		#device_value = -1 



	#load data
	print("Load data...")
	text_field = data.Field(lower = True, batch_first = True, fix_length = 15)
	label_field = data.Field(sequential = False)

	#device = - 1 : cpu 
	train_loader, dev_loader, test_loader = SST_data_loader(text_field, label_field, glove, batch_size, device = device_value, repeat = False)

    #parameters 
	in_channels = 1 
	out_channels = 2
	voca_size = len(text_field.vocab)
	num_classes = len(label_field.vocab) - 1 
	embed_dim = glove.vectors.size()[1]
	kernel_sizes = [2,3,4]
	dropout_p = 0.5 
	embedding_weight = text_field.vocab.vectors

	learnign_rate = 0.001
	num_epochs = 100


	# model 
	cnn = model.CNNClassifier(in_channels, out_channels, voca_size, embed_dim, num_classes, kernel_sizes, dropout_p, embedding_weight)

	if iscuda:
		cnn = cnn.cuda()

	# train 
	train.train(train_loader, test_loader, cnn, iscuda, learnign_rate, num_epochs)


	