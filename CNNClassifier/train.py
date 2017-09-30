import os 
import sys
import torch 
import torch.autograd as autograd
import torch.nn as nn

def train(train_loader, test_loader, model, cuda, learnign_rate, num_epochs):

    # gpu runnable 
	if cuda: 
		model.cuda()
    
    # train mode 
	model.train()

	#Loss and optimizer 
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr = learnign_rate)

	for epoch in range(num_epochs):
		for i, batch in enumerate(train_loader):
			feature, target = batch.text, batch.label
			target.data.sub_(1)  # index 
			if cuda:
				feature, target = feature.cuda(), target.cuda()

			#Forward, Backward, Optimize 
			optimizer.zero_grad()
			output = model(feature)
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()

			if(i+1) % 100 == 0:
				print ('Epoch [%d/%d], Loss: %.4f'  
					%(epoch+1, num_epochs,  loss.data[0]))


 

