import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable

class CNNClassifier(nn.Module):
	def __init__(self, in_channels, out_channels, voca_size, embed_dim, num_classes, kernel_sizes, dropout_p, embedding_weight):
		super(CNNClassifier, self).__init__()
		self.embedding_weight = embedding_weight
		self.embedding = nn.Embedding(voca_size, embed_dim)
		self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels,(k_size, embed_dim)) for k_size in kernel_sizes])

		self.dropout = nn.Dropout(dropout_p) 
		self.fc = nn.Linear(len(kernel_sizes) * out_channels , num_classes)
		self.init_weights()

	def init_weights(self):
		self.embedding.weight = nn.Parameter(self.embedding_weight)
		self.fc.bias.data.normal_(0, 0.01)
		self.fc.weight.data.normal_(0, 0.01)

		for layer in self.convs:
			nn.init.xavier_normal(layer.weight)


	def forward(self, x):
		
		"""
		parameters of x:
		                N: batch_size 
		                C: num of in_channels
		                W: len of text 
		                D: num of embed_dim 
		"""

		x = self.embedding(x) # (N,W,D)
		#x= Variable(x)
		x = x.unsqueeze(1) #(N,1,W,D)
		x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
		x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
		x = torch.cat(x,1)

		x = self.dropout(x)
		out = self.fc(x)

		return out 


class RNNClassifier(nn.Module):
	def __init__(self, voca_size, embed_size, hidden_size, num_layers, num_classes, embedding_weight):
		super(RNNClassifier,self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.embedding_weight = embedding_weight
		self.embed = nn.Embedding(voca_size, embed_size)
		self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first= True)
		self.dropout = nn.Dropout(0.5)
		self.fc = nn.Linear(hidden_size, num_classes)
		self.init_weights()


	def init_weights(self):

		self.embed.weight = nn.Parameter(self.embedding_weight)
		self.fc.bias.data.normal_(0, 0.01)
		self.fc.weight.data.normal_(0, 0.01)
	

	def forward(self, x):

		x = self.embed(x)

		# Set initial states  & GPU run
		h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
		c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()

		h0 = (nn.init.xavier_normal(h0))
		c0 = (nn.init.xavier_normal(c0))

		# Forward 
		out, _ = self.lstm(x, (h0,c0))
        # Decode hidden state of last time step/ many to one 
		out = self.dropout(out)
		out = self.fc(out[:, -1, :])
		return out

		
class RCNN_Classifier(nn.Module):
	def __init__(self, voca_size, embed_size, hidden_size, sm_hidden_size,  num_layers, num_classes, embedding_weight):
		super(RCNN_Classifier,self).__init__()
		self.hidden_size = hidden_size
		self.sm_hidden_size = sm_hidden_size
		self.num_layers = num_layers

		self.embedding_weight = embedding_weight
		self.embed = nn.Embedding(voca_size, embed_size)
		self.bi_lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first= True, bidirectional = True)
		self.sm_fc = nn.Linear(embed_size + hidden_size*2 , sm_hidden_size)

		self.fc = nn.Linear(sm_hidden_size, num_classes)
		self.init_weights()

	def init_weights(self):

		self.embed.weight = nn.Parameter(self.embedding_weight)

	def forward(self, x):
		x = self.embed(x)

		#Set inital states & GPU run
		h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)).cuda()
		c0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)).cuda() # *2 for bidirectional

		h0 = (nn.init.xavier_normal(h0))
		c0 = (nn.init.xavier_normal(c0))

		#Forward 
		lstm_out, _ = self.bi_lstm(x, (h0, c0))
		out = torch.cat((lstm_out, x), 2)  # eq. 3 

		y2 = F.tanh(self.sm_fc(out)) # semantic layer eq.4  y2
		y2 = y2.unsqueeze(1)

		y3 = F.max_pool2d(y2, (y2.size(2),1)).squeeze() # eq.5  y3

		y4 = self.fc(y3) # eq.6

		final_out = F.softmax(y4) # eq.7

		return final_out
		