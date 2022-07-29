# Core libraries
import os
import csv
from sklearn import datasets
import torch
import argparse
import numpy as np
from tqdm import tqdm
from utilities.ioutils import *
from sklearn.neighbors import KNeighborsClassifier

# PyTorch
import torch
from torch.utils import data
from torch.autograd import Variable

# Local libraries
from utilities.utils import Utilities
from models.embeddings import resnet50

# Import our dataset class

from datasets.Pipeline.Pipeline import Pipeline


def load_class_dict_csv(file):
	class_names = {}
	try:
		with open(file, 'r') as f:
			w= csv.DictReader(f,delimiter=',')
			line_count = 0
			lines = list(w)
			for row in lines:
				class_names[row["class"]] = row["animal_id"]
	except IOError:
		print("I/O error")
	
	return class_names

def get_predictions(data_loader, num_classes):
	# Define our embeddings model
	model = resnet50(pretrained=True, num_classes=num_classes, ckpt_path=args.model_path, embedding_size=args.embedding_size)
	
	# Put the model on the GPU and in evaluation mode
	model.cuda()
	model.eval()

	# Embeddings/labels to be stored on the testing set
	outputs_embedding = np.zeros((1,args.embedding_size))
	filenames = []
	for images, filename in tqdm(data_loader, desc=f"Inferring  embeddings"):
		# Put the images on the GPU and express them as PyTorch variables
		
		images = Variable(images.cuda())

		# Get the embeddings of this batch of images
		outputs = model(images)

		# Express embeddings in numpy form
		embeddings = outputs.data
		embeddings = embeddings.cpu().numpy()

		# Store testing data on this batch ready to be evaluated
		
		outputs_embedding = np.concatenate((outputs_embedding,embeddings), axis=0)
		
		for f in filename:
			filenames.append(f)
		
	outputs_embedding = outputs_embedding[1:]
	return outputs_embedding, filenames
	
# For a trained model, let's evaluate it
def clusterData(args):
	
	# Get the embeddings and labels of the training set and testing set
	training_data = np.load(args.train_embeddings)
	train_embeddings, train_labels = training_data["embeddings"][1:],training_data["labels"][1:]
	
	num_classes = np.unique(train_labels).shape[0]
	classes = load_class_dict_csv(args.class_labels)

	dataset = Pipeline(args.input,transform=True)
	data_loader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=6, shuffle=False)
	outputs_embedding, filenames = get_predictions(data_loader, num_classes)

	# Classify them
	neigh = KNeighborsClassifier(n_neighbors=args.n_neighbours, n_jobs=-4)

	# Give it the embeddings and labels of the training set
	neigh.fit(train_embeddings, train_labels)

	# Get the predictions from KNN
	predictions = neigh.predict(outputs_embedding)
	prediction_id = [classes[str(int(i))] for i in predictions]
	output_csvfile = os.path.join(args.save_path,"Predictions.csv")
	print("Writing predictions to csv file:"+str(output_csvfile))
	with open(output_csvfile, 'w', newline='') as f:
		writer = csv.writer(f)
		for p in range(len(predictions)):
			writer.writerow([filenames[p],prediction_id[p]])
		

	if args.save_embeddings:
		print("Saving Data Embeddings at locations:"+str(args.save_path))
		save_path = os.path.join(args.save_path, "data_embeddings.npz")
		
		# Save the embeddings to a numpy array
		np.savez(save_path,  embeddings=outputs_embedding, labels=predictions)

# Main/entry method
if __name__ == '__main__':
	# Collate command line arguments
	parser = argparse.ArgumentParser(description='Params')

	# Required arguments
	parser.add_argument('--model_path', nargs='?', type=str, required=True, 
						help='Path to the saved model to load weights from')
	parser.add_argument('--save_path', type=str, required=True,
						help="Where to store the embeddings and output")
	parser.add_argument('--embedding_size', nargs='?', type=int, default=128, 
						help='Size of the dense layer for inference')
	parser.add_argument('--batch_size', nargs='?', type=int, default=16,
						help='Batch Size')
	parser.add_argument('--save_embeddings', type=bool, default=True,
						help="Should we save the embeddings to file")
	parser.add_argument('--n_neighbours', type=int, default=1,
						help="Number of neightbours used in KNN")
	parser.add_argument('--train_embeddings', type=str, required=True,
						help="")				
	parser.add_argument('--input', type=str, required=True,
						help="")
	parser.add_argument('--class_labels', type=str, required=True,
						help="")	
	args = parser.parse_args()

	# Let's infer some embeddings
	clusterData(args)
