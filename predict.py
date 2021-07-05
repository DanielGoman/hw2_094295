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

import argparse

from dataset import FacemaskRecognitionDataset, dir_to_df
from MyModel import get_model
from train import get_transform, collate_fn, get_gpu
from evaluation import calc_iou



# paths
model_path = 'model'
predictions_path = 'prediction.csv'


def predict(test_path, model=None, save=True):
	#Test Dataset
	test_df = dir_to_df(test_path)
	test_dataset = FacemaskRecognitionDataset(test_df, test_path, mode = 'test', transforms = get_transform())

	#Test data loader
	test_loader = DataLoader(
	    test_dataset,
	    batch_size=1,
	    shuffle=False,
	    num_workers=1,
	    drop_last=False,
	    collate_fn=collate_fn
	)



	prediction_df = pd.DataFrame(columns=['filename', 'x', 'y', 'w', 'h', 'proper_mask'])

	device = get_gpu()


	if model is None:
		model = get_model('fasterrcnn_resnet')
		#model.load_state_dict(torch.load(f'{model_path}_fasterrcnn_resnet.pkl'))
		model.load_state_dict(torch.load(f'{model_path}.pkl'))

	model.eval()
	model.to(device)

	threshold = 0.5


	print('Evaluating')
	iou_scores = []
	acc_scores = []
	i = 0
	true_count = 0
	false_count = 0
	for idx, (images, image_names) in enumerate(test_loader):
	    #Forward ->
	    images = list(image.to(device) for image in images)
	    with torch.no_grad():
	        output = model(images)

	    #Converting tensors to array
	    out_boxes = output[0]['boxes'].data.cpu().numpy()
	    out_scores = output[0]['scores'].data.cpu().numpy()
	    out_labels = output[0]['labels'].data.cpu().numpy()


	    if len(out_boxes) > 0:
	        boxes = out_boxes[0]
	        label = 'True' if out_scores[0] > threshold else 'False'
	    else:
	        # guess
	        boxes = [0.25 * images[0].size()[1],
	                 0.35 * images[0].size()[2],
	                 0.75 * images[0].size()[1],
	                 0.65 * images[0].size()[2]]
	        label = 'False'


	    x = boxes[0]
	    y = boxes[1]
	    w = boxes[2] - x
	    h = boxes[3] - y

	    #Creating row for df
	    row = {"filename" : image_names[0], "x" : x, "y" : y, "w" : w, "h" : h, "proper_mask" : label}
	    
	    #Appending to df
	    prediction_df = prediction_df.append(row, ignore_index = True)

	    pred_box = [x, y, w, h]
	    true_box = [int(val) for val in image_names[0].split('__')[1][1:-1].split(',')]
	    curr_iou_score = calc_iou(pred_box, true_box)
	    iou_scores.append(curr_iou_score)

	    true_label = image_names[0].split('__')[2].split('.')[0]
	    acc_scores.append(label == true_label)

	    if idx % 500 == 0 and idx > 0:
	    	print(f'Completed {idx}')


	iou_score = np.mean(iou_scores)
	acc_score = np.mean(acc_scores)
	print(f'IoU score: {iou_score}')
	print(f'acc_score: {acc_score}')

	print(f'\nWriting predictions to: {predictions_path}')

	if save:
		prediction_df.to_csv(predictions_path, index = False)

	return acc_score, iou_score



if __name__=="__main__":
	# Parsing script arguments
	parser = argparse.ArgumentParser(description='Process input')
	parser.add_argument('test_path', type=str, help='test directory path')
	args = parser.parse_args()
	predict(args.test_path)




