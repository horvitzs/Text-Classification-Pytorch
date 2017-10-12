import torch.nn as nn 
import torch 
from torch.autograd import Variable


class RNNClassifier(nn.Module):
	def __init__(self, voca_size, embed_size, hidden_size, num_layers, num_classes, embedding_weight):
		super(RNNClassifier,self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.embedding_weight = embedding_weight
		self.embed = nn.Embedding(voca_size, embed_size)
		self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first= True)
		self.fc = nn.Linear(hidden_size, num_classes)
		self.init_weights()


	def init_weights(self):

		self.embed.weight = nn.Parameter(self.embedding_weight)
		#self.fc.bias.data.fill_(0)
		#self.fc.weight.data.fill_(-0.1, 0.1)


	def forward(self, x):

		x = self.embed(x)

		# Set initial states 
		h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
		c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

		# Forward 
		out, _ = self.lstm(x, (h0,c0))
        # Decode hidden state of last time step/ many to one 
		out = self.fc(out[:, -1, :])
		return out

		
