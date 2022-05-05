# Core libraries
import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

# PyTorch
import torch
from torch.utils import data
from torch.autograd import Variable

# Local libraries
from utilities.utils import Utilities
from models.embeddings import resnet50

# Import our dataset class
from datasets.OpenSetCows2020.OpenSetCows2020 import OpenSetCows2020

"""
File for inferring the embeddings of the validation portion of a selected database and
evaluating its classification performance using KNN

"""
# example command python evaluation.py --model_path=/user/work/gh18931/diss/MetricLearningIdentification/output/output6/aug_validation/fold_0/best_model_state.pkl --dataset=Zebra2 --folds_file=/user/work/gh18931/diss/datasets/grevvy_dataset/splits/folds.json --save_path=./

# For a trained model, let's evaluate it
def evaluateModel(args):
	# Load the relevant datasets
	train_dataset = Utilities.selectDataset(args, True)
	validation_dataset = Utilities.selectDataset(args, False)

	# Get the embeddings and labels of the training set and validation set
	
	train_embeddings, train_labels = inferEmbeddings(args, train_dataset, "train")
	validation_embeddings, validation_labels = inferEmbeddings(args, validation_dataset, "test")
	
	validation_embeddings = validation_embeddings[1:]
	validation_labels = validation_labels[1:]
	# Classify them
	
	accuracy = KNNAccuracy(train_embeddings, train_labels, validation_embeddings, validation_labels,args.n_neighbours)
	KNNClassAccuracy(train_embeddings, train_labels, validation_embeddings, validation_labels,args.n_neighbours)

	# Write it out to the console so that subprocess can pick them up and close
	print(f"Accuracy={str(accuracy)}")

def KNNClassAccuracy(train_embeddings, train_labels,validation_embeddings, validation_labels, n_neighbors=5):
	 # Define the KNN classifier

	neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-4,weights="distance")

	# Give it the embeddings and labels of the training set
	neigh.fit(train_embeddings, train_labels)

	# Total number of validation instances
	#total = len(validation_labels-1)

	# Get the predictions from KNN
	predictions = neigh.predict(validation_embeddings)
	prediction_prob = neigh.predict_proba(validation_embeddings)
	# Set of unique labels
	uniqueLabels = set(validation_labels)

	for validation_label in uniqueLabels:
		
		correct = torch.tensor((predictions == validation_labels) * (validation_labels == validation_label)).float().sum()
		total = max((validation_labels == validation_label).sum(),1)
		
	# Compute accuracy

		accuracy = (float(correct) / total) * 100
		print(f"Class : {str(validation_label)} , Accuracy = {str(accuracy)}")
	
def KNNAccuracy_N(train_embeddings, train_labels,validation_embeddings, validation_labels, n_neighbors=3,n_value =1):
	# Define the KNN classifier
	neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-4,weights="distance")

	# Give it the embeddings and labels of the training set
	neigh.fit(train_embeddings, train_labels)

	# Total number of validation instances
	total = len(validation_labels-1)

	# Get the predictions from KNN

	
	predictions = neigh.predict(validation_embeddings)

	prediction_prob = neigh.predict_proba(validation_embeddings)
	
	prediction_top_n_ind = []
	for i in prediction_prob:
		prediction_top_n_ind.append(list((-i).argsort()[:n_value]))
	# How many were correct?
	
	n_correct = 0
	for i, pred in enumerate(prediction_top_n_ind):

		if validation_labels[i] in pred:
			n_correct +=1

	correct = (predictions == validation_labels).sum()
	
	# Compute accuracy
	accuracy = (float(correct) / total) * 100
	n_accuracy = (float(n_correct)/total) * 100

	print(f"N-Accuracy:{n_accuracy}")

	return accuracy
   
# Use KNN to classify the embedding space
def KNNAccuracy(train_embeddings, train_labels,validation_embeddings, validation_labels, n_neighbors=3):
	# Define the KNN classifier
	neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-4,weights="distance")

	# Give it the embeddings and labels of the training set
	neigh.fit(train_embeddings, train_labels)

	# Total number of validation instances
	total = len(validation_labels-1)

	# Get the predictions from KNN
	predictions = neigh.predict(validation_embeddings)
	prediction_prob = neigh.predict_proba(validation_embeddings)
	
	# How many were correct?
	correct = (predictions == validation_labels).sum()
	
	# Compute accuracy
	accuracy = (float(correct) / total) * 100

	return accuracy

# Infer the embeddings for a given dataset
def inferEmbeddings(args, dataset, split):
	# Wrap up the dataset in a PyTorch dataset loader
	data_loader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=6, shuffle=False)

	# Define our embeddings model
	model = resnet50(pretrained=True, num_classes=dataset.getNumClasses(), ckpt_path=args.model_path, embedding_size=args.embedding_size)
	
	# Put the model on the GPU and in evaluation mode
	model.cuda()
	model.eval()

	# Embeddings/labels to be stored on the validation set
	outputs_embedding = np.zeros((1,args.embedding_size))
	labels_embedding = np.zeros((1))
	total = 0
	correct = 0

	# Iterate through the validation portion of the dataset and get
	for images, _, _, labels, _ in tqdm(data_loader, desc=f"Inferring {split} embeddings"):
		# Put the images on the GPU and express them as PyTorch variables
		images = Variable(images.cuda())

		# Get the embeddings of this batch of images
		outputs = model(images)

		# Express embeddings in numpy form
		embeddings = outputs.data
		embeddings = embeddings.cpu().numpy()

		# Convert labels to readable numpy form
		labels = labels.view(len(labels))
		labels = labels.cpu().numpy()
		
		# Store validation data on this batch ready to be evaluated
		outputs_embedding = np.concatenate((outputs_embedding,embeddings), axis=0)
		labels_embedding = np.concatenate((labels_embedding,labels), axis=0)
	
	# If we're supposed to be saving the embeddings and labels to file
	if args.save_embeddings:
		pass
		# Construct the save path
		#save_path = os.path.join(args.save_path, f"{split}_embeddings.npz")
		
		# Save the embeddings to a numpy array
		#np.savez(save_path,  embeddings=outputs_embedding, labels=labels_embedding)

	return outputs_embedding, labels_embedding

# Main/entry method
if __name__ == '__main__':
	# Collate command line arguments
	parser = argparse.ArgumentParser(description='Params')

	# Required arguments
	parser.add_argument('--model_path', nargs='?', type=str, required=True, 
						help='Path to the saved model to load weights from')
	parser.add_argument('--folds_file', type=str, default="", required=True,
						help="The file containing known/unknown splits")
	parser.add_argument('--save_path', type=str, required=True,
						help="Where to store the embeddings")

	parser.add_argument('--dataset', nargs='?', type=str, default='Zebra', 
						help='Which dataset to use')
	parser.add_argument('--batch_size', nargs='?', type=int, default=16,
						help='Batch Size')
	parser.add_argument('--embedding_size', nargs='?', type=int, default=128, 
						help='Size of the dense layer for inference')
	parser.add_argument('--current_fold', type=int, default=0,
						help="The current fold we'd like to test on")
	parser.add_argument('--save_embeddings', type=bool, default=True,
						help="Should we save the embeddings to file")
	parser.add_argument('--n_neighbours', type=int, default=5,
					help="Number of neightbours used in KNN")					
	args = parser.parse_args()

	# Let's infer some embeddings
	evaluateModel(args)

[10, 10, 27, 19, 12, 12, 13, 31, 13, 14, 15,  8, 12, 16, 32, 17,  2,  2,34, 19,  2, 15, 11, 20, 22, 29, 22, 20, 23, 22, 24, 11,  6, 25, 26, 33,27, 27, 11, 29, 29, 29, 29, 22, 30, 37, 21, 31, 32, 32, 33, 33, 34, 34, 35, 11, 36, 36, 37, 37, 11, 19, 22,  4,  5, 34,  3, 29,  7, 16, 29, 22,  9,  9]

[10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19,  2,  2, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29,  3,  3, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34,35, 35, 36, 36, 37, 37, 38, 38,  4,  4,  5,  5,  6,  6,  7,  7,  8,  8,  9,  9,]