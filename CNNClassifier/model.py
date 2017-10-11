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

		
