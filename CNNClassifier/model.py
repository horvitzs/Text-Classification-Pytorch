import torch 
import torch.nn as nn 
import torch.nn.functional as F 

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
		x = x.unsqueeze(1) #(N,1,W,D)
		x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
		x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
		x = torch.cat(x,1)

		x = self.dropout(x)
		logit = self.fc(x)

		return logit 

