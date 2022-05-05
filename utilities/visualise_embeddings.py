# Core libraries
import os
import sys
import cv2
import argparse
import numpy as np

# Matplotlib / TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.patheffects as PathEffects

# Load and visualise test and train embeddings via t-SNE together
def plotAllEmbeddings(args):
	if not os.path.exists(args.embeddings_file):
		print(f"No embeddings file at path: {args.embeddings_file}, exiting.")
		sys.exit(1)
	# Load the embeddings into memory
	train_embeddings = np.load(args.embeddings_file)
	test_name = str(args.embeddings_file).replace("train","test")
	test_embeddings = np.load(test_name)
	embedding = np.concatenate((train_embeddings["embeddings"],test_embeddings["embeddings"]),axis =0)
	label = np.concatenate((train_embeddings["labels"],test_embeddings["labels"]),axis=0)
	retained_label = [-i for i in test_embeddings["labels"][1:]]
	label_split = np.concatenate((train_embeddings["labels"],retained_label),axis=0)
	embeddings = {"embeddings":embedding,"labels":label,"labels_split":label_split}

	# Visualise the learned embedding via t-SNE
	
	visualiser = TSNE(n_components=2, perplexity=args.perplexity,n_iter=2500,init="pca")

	# Reduce dimensionality
	
	reduction = visualiser.fit_transform(embeddings['embeddings'])

	print("Visualisation computed")
	filename=  os.path.basename(args.embeddings_file)[:-4]+f"complete_preplexity-{args.perplexity}"
	# Plot the results and save to file
	scatterAll(reduction, embeddings['labels_split'], filename)

def scatterAll(x, labels, filename):
	x = x[1:]
	scatter_x = x[:,0]
	scatter_y = x[:,1]
	labels = labels[1:]

	# Get the number of classes (number of unique labels)
	num_classes = np.unique(labels).shape[0] //2
	#group = np.unique(labels)
	cdict = {1: 'red', 2: 'blue', 3: 'green',4: 'dodgerblue', 5: 'beige', 6: 'brown',7: 'coral', 8: 'cyan', 9:'fuchsia',10: 'gold', 11: 'grey', 12: 'indigo',13: 'mediumpurple', 14: 'lightblue', 15: 'lime',16: 'magenta', 17: 'maroon', 18: 'navy',19: 'olive', 20: 'orange', 21: 'orchid',22: 'pink', 23: 'crimson', 24: 'purple',25: 'salmon', 26: 'silver', 27: 'tan',28: 'teal', 29: 'chocolate', 30: 'turquoise',31: 'violet', 32: 'wheat', 33: 'yellow',34: 'lightgreen', 35: 'darkgreen', 36: 'black', 37: 'yellowgreen', 38: 'goldenrod'}
	
	fig, ax = plt.subplots(figsize=(9,9))
	
	for g in np.unique(labels):

		if g < 0:
			ix = np.where(labels == g)
			ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[int(-g)],alpha=1.0, label =  f"Class {int(-g)}",edgecolors='black',linewidths=1, s = 40)
		else:
			ix = np.where(labels == g)
			ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[int(g)],alpha=0.2,s = 40)
	ax.legend(bbox_to_anchor=(1.04,1.1), loc="upper left")
	

	plt.axis('off')
	plt.savefig(f"test_.png", bbox_inches='tight',pad_inches = 0)
	

def scatter(x, labels, filename):
	x = x[1:]
	scatter_x = x[:,0]
	scatter_y = x[:,1]
	labels = labels[1:]
	print(len(labels))
	# Get the number of classes (number of unique labels)
	num_classes = np.unique(labels).shape[0]

	cdict = {1: 'red', 2: 'blue', 3: 'green',4: 'dodgerblue', 5: 'beige', 6: 'brown',7: 'coral', 8: 'cyan', 9:'fuchsia',10: 'gold', 11: 'grey', 12: 'indigo',13: 'mediumpurple', 14: 'lightblue', 15: 'lime',16: 'magenta', 17: 'maroon', 18: 'navy',19: 'olive', 20: 'orange', 21: 'orchid',22: 'pink', 23: 'crimson', 24: 'purple',25: 'salmon', 26: 'silver', 27: 'tan',28: 'teal', 29: 'chocolate', 30: 'turquoise',31: 'violet', 32: 'wheat', 33: 'yellow',34: 'lightgreen', 35: 'darkgreen', 36: 'black', 37: 'yellowgreen', 38: 'goldenrod'}
	
	fig, ax = plt.subplots(figsize=(9,9))
	
	for g in np.unique(labels):
		ix = np.where(labels == g)
		ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[int(g)], label =  f"Class {int(g)}", s = 100)
	ax.legend(bbox_to_anchor=(1.04,1.1), loc="upper left")
	
	plt.axis('off')
	plt.savefig(filename+".png", bbox_inches='tight',pad_inches = 0)
	


# Load and visualise embeddings via t-SNE
def plotEmbeddings(args):
	# Ensure there's something there
	if not os.path.exists(args.embeddings_file):
		print(f"No embeddings file at path: {args.embeddings_file}, exiting.")
		sys.exit(1)

	# Load the embeddings into memory
	embeddings = np.load(args.embeddings_file)

	print("Loaded embeddings")

	# Visualise the learned embedding via t-SNE
	visualiser = TSNE(n_components=2, perplexity=args.perplexity,init='pca')

	# Reduce dimensionality
	reduction = visualiser.fit_transform(embeddings['embeddings'])

	print("Visualisation computed")
	filename=  os.path.basename(args.embeddings_file)[:-4]+f"_test_preplexity-{args.perplexity}"
	# Plot the results and save to file
	scatter(reduction, embeddings['labels'], filename)

# Main/entry method
if __name__ == '__main__':
	# Collate command line arguments
	parser = argparse.ArgumentParser(description='Parameters for visualising the embeddings via TSNE')
	parser.add_argument('--embeddings_file', type=str, required=True,
						help="Path to embeddings .npz file you want to visalise")
	parser.add_argument('--perplexity', type=int, default=30,
						help="Perplexity parameter for t-SNE, consider values between 5 and 50")
	args = parser.parse_args()

	# Let's plot!
	plotAllEmbeddings(args)
	#plotEmbeddings(args)
	