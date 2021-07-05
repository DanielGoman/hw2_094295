import os
import cv2
import time
import pandas as pd
import numpy as np

from PIL import Image

import torch
import torchvision
import torchvision.transforms as T

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

from dataset import FacemaskRecognitionDataset, dir_to_df
from MyModel import get_model

# paths
train_path = "/home/student/train/"
model_path = 'model'

# hyperparameters
epochs = 5
lr = 0.005
batch_size = 2

models = ['fasterrcnn_resnet']#, 'mobilenet', 'mobilenet_320']

# util functions
def get_transform():
    return T.Compose([T.ToTensor(), ])

def collate_fn(batch):
    return tuple(zip(*batch))

def get_gpu():
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	torch.cuda.empty_cache()

	return device


def main():
	from predict import predict

	train_df = dir_to_df(train_path)

	train_dataset = FacemaskRecognitionDataset(train_df, train_path, transforms = get_transform())

	train_loader = DataLoader(
	    train_dataset,
	    batch_size = batch_size,
	    shuffle = True,
	    num_workers = 2,
	    collate_fn = collate_fn
	)
	print('Created dataset')

	device = get_gpu()

	for model_name in models:
		model = get_model(model_name)

		#Retriving all trainable parameters from model (for optimizer)
		params = [p for p in model.parameters() if p.requires_grad]

		#Defininig Optimizer

		#optimizer = torch.optim.Adam(params, lr = lr)
		optimizer = torch.optim.SGD(params, lr = lr, momentum = 0.9)

		model.to(device)


		print('Starting train')
		itr = 1
		total_train_loss = []

		acc_scores = []
		iou_scores = []

		for epoch in range(epochs):
		    model.train()

		    start_time = time.time()
		    train_loss = []
		    
		    start = time.time()
		    #Retriving Mini-batch
		    for images, targets, image_names in train_loader:

		        #Loading images & targets on device
		        images = list(image.to(device) for image in images)
		        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
		        
		        #Forward propagation
		        out = model(images, targets)
		        losses = sum(loss for loss in out.values())
		        
		        #Reseting Gradients
		        optimizer.zero_grad()
		        
		        #Back propagation
		        losses.backward()
		        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
		        optimizer.step()
		        
		        #Average loss
		        loss_value = losses.item()
		        train_loss.append(loss_value)
		        
		        if itr % 1000 == 0:
		            print(f"\n Iteration #{itr} loss: {out} \n")
		            print(f'elapsed time: {time.time() - start}')
		            start = time.time()

		        itr += 1
		    
	 	
		    time_elapsed = time.time() - start_time
		    print("Time elapsed: ",time_elapsed)

		    # getting train scores - loss, acc, iou
		    print(f'Evaluating model\'s scores\n')
		    epoch_train_loss = np.mean(train_loss)
		    total_train_loss.append(epoch_train_loss)

		    start = time.time()
		    acc_score, iou_score = predict(train_path, model=model, save=False)
		    acc_scores.append(acc_score)
		    iou_scores.append(iou_score)
		    print(f'Epoch {epoch} train loss is {epoch_train_loss:.4f}')
		    print(f'Epoch {epoch} train accuracy is {acc_score:.4f}')
		    print(f'Epoch {epoch} train IoU is {iou_score:.4f}')
		    print(f'Evaluation of epoch took {time.time() - start}\n')

		# save model
		torch.save(model.state_dict(), f'{model_path}_{model_name}.pkl')

		plot_scores(total_train_loss, 'Loss', 'Train loss', f'train_loss_{model_name}', epochs)
		plot_scores(acc_scores, 'Accuracy', 'Train accuracy', f'train_accuracy_{model_name}', epochs)
		plot_scores(iou_scores, 'IoU', 'Train IoU', f'train_iou_{model_name}', epochs)



def plot_scores(data, ylabel, title, file_name, num_epochs):
	fig, ax = plt.subplots()
	ax.plot(np.arange(num_epochs) + 1, data)
	plt.xlabel('Epochs')
	plt.ylabel(ylabel)
	plt.title(title)
	plt.savefig(f'{file_name}.jpg')


if __name__=="__main__":
	main()