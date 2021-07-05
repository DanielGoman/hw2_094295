
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(model_type='fasterrcnn_resnet'):
	if model_type == 'fasterrcnn_resnet':
	    #Faster - RCNN Model - not pretrained
	    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
	    
	elif model_type == 'mobilenet':
	    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False)

	elif model_type == 'mobilenet_320':
	    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False)

	num_classes = 2

	# get number of input features for the classifier
	in_features = model.roi_heads.box_predictor.cls_score.in_features

	# replace the pre-trained head with a new one
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

	return model