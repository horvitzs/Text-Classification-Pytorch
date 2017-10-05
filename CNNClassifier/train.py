import os 
import sys
import torch 
import torch.autograd as autograd
import torch.nn as nn

def train(train_loader, dev_loader, model, cuda, learnign_rate, num_epochs):

    # gpu runnable 
	if cuda: 
		model.cuda()
    
    # train mode 
	model.train()

	num_batch = len(train_loader.dataset)
	step = 0

	#Loss and optimizer 
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr = learnign_rate)

	for epoch in range(num_epochs):
		for i, batch in enumerate(train_loader):
			feature, target = batch.text, batch.label
			target.data.sub_(1)  # index 
			if cuda:
				feature, target = feature.cuda(), target.cuda()
			#if feature.size()[1] < 10:
				#continue 

			#Forward, Backward, Optimize 
			optimizer.zero_grad()
			output = model(feature)
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()

			step += 1 



			if(i+1) % 100 == 0:
				print ('Epoch [%d/%d], Steps [%d/%d], Loss: %.4f'  
					%(epoch+1, num_epochs, step,  num_batch * num_epochs, loss.data[0]))

			if(step) % 1000 == 0:
				eval(dev_loader, model, cuda)



def eval(test_loader, model, cuda):
 	#eval mode 
 	model.eval()

 	#Loss and optimizer 
 	criterion = nn.CrossEntropyLoss()

 	corrects = 0
 	avg_loss = 0 

 	for i, batch in enumerate(test_loader):
 		feature, target = batch.text, batch.label
 		target.data.sub_(1) # index
 		if cuda:
 			feature, target = feature.cuda(), target.cuda()

 		output = model(feature)
 		loss = criterion(output, target) # losses are summed, not average 

 		avg_loss += loss.data[0]
 		corrects += (torch.max(output, 1)
                     [1].view(target.size()).data == target.data).sum()
 	
 	size = len(test_loader.dataset)
 	accuracy = 100 * corrects / size 
 	model.train()
 	print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))



def predict(sample_text, model, text_field, label_field):

	text = text_field.preprocess(sample_text)
	text = [[text_field.vocab.stoi[x] for x in text]]
	x = text_field.tensor_type(text)
	x = autograd.Variable(x)
	x = x.cuda()

	output = model(x)
	_, predicted = torch.max(output, 1)

	return label_feild.vocab.itos[predicted.data[0][0]+1]








