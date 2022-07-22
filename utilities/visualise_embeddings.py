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

	test_name = str(args.embeddings_file).replace("train","test")
	train_embeddings = np.load(args.embeddings_file)
	test_embeddings = np.load(test_name)

	embedding = np.concatenate((train_embeddings["embeddings"],test_embeddings["embeddings"]),axis =0)
	

	# negates the integer class of test labels to differentaiate them from training 

	test_label = [-i for i in test_embeddings["labels"][1:]]
	label_split = np.concatenate((train_embeddings["labels"],test_label),axis=0)
	embeddings = {"embeddings":embedding,"labels":label_split}

	# Visualise the learned embedding via t-SNE
	
	visualiser = TSNE(n_components=2, perplexity=args.perplexity,n_iter=2500,init="pca")

	# Reduce dimensionality
	
	reduction = visualiser.fit_transform(embeddings['embeddings'])

	print("Visualisation computed")
	filename =  f"complete_scatter_preplexity-{args.perplexity}"
	# Plot the results and save to file

	scatterAll(reduction, embeddings['labels'], filename)

def scatterAll(x, labels, filename):
	# Removes blank entry
	x = x[1:]
	labels = labels[1:]


	scatter_x = x[:,0]
	scatter_y = x[:,1]
	

	# Colour dictionary - ensures each class ( for 38 classes) has a unqiue colour
	cdict = {1: 'red', 2: 'blue', 3: 'green',4: 'dodgerblue', 5: 'seagreen', 6: 'brown',7: 'coral', 8: 'cyan', 9:'fuchsia',10: 'gold', 11: 'grey', 12: 'indigo',13: 'mediumpurple', 14: 'lightblue', 15: 'lime',16: 'magenta', 17: 'maroon', 18: 'navy',19: 'olive', 20: 'orange', 21: 'orchid',22: 'darkslateblue', 23: 'crimson', 24: 'purple',25: 'salmon', 26: 'silver', 27: 'tan',28: 'teal', 29: 'chocolate', 30: 'turquoise',31: 'violet', 32: 'steelblue', 33: 'yellow',34: 'lightgreen', 35: 'darkgreen', 36: 'black', 37: 'yellowgreen', 38: 'goldenrod'}
	
	fig, ax = plt.subplots(figsize=(9,9))
	# Shades points differently if from the train data or test data
	for g in np.unique(labels):

		if g < 0:
			ix = np.where(labels == g)
			ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[int(-g)],alpha=1.0, label =  f"Class {int(-g)-1}",edgecolors='black',linewidths=1.5, s = 40)
		else:
			ix = np.where(labels == g)
			ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[int(g)],alpha=0.15,s = 40)
	
	
	# Formatting
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(reversed(handles), reversed(labels),bbox_to_anchor=(1.04,1.1), loc="upper left")
	plt.axis('off')

	# Save it to file
	plt.savefig(filename, bbox_inches='tight',pad_inches = 0)


# Define our own plot function
def scatter(x, labels, filename):
	# Get the number of classes (number of unique labels)
	num_classes = np.unique(labels).shape[0]

	# Choose a color palette with seaborn.
	palette = np.array(sns.color_palette("hls", num_classes+1))

	# Map the colours to different labels
	label_colours = np.array([palette[int(labels[i])] for i in range(labels.shape[0])])

	# Create our figure/plot
	f = plt.figure(figsize=(8, 8))
	ax = plt.subplot(aspect='equal')

	# Plot the points
	ax.scatter(	x[:,0], x[:,1], 
				lw=0, s=40, 
				c=label_colours, 
				marker="o")

	# Do some formatting
	plt.xlim(-25, 25)
	plt.ylim(-25, 25)
	ax.axis('off')
	ax.axis('tight')
	plt.tight_layout()

	# Save it to file
	plt.savefig(filename+".pdf")
	


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
	visualiser = TSNE(n_components=2, perplexity=args.perplexity)

	# Reduce dimensionality
	reduction = visualiser.fit_transform(embeddings['embeddings'])

	print("Visualisation computed")

	# Plot the results and save to file
	scatter(reduction, embeddings['labels'], os.path.basename(args.embeddings_file)[:-4])

# Main/entry method
if __name__ == '__main__':
	# Collate command line arguments
	parser = argparse.ArgumentParser(description='Parameters for visualising the embeddings via TSNE')
	parser.add_argument('--embeddings_file', type=str, required=True,
						help="Path to embeddings .npz file you want to visalise")
	parser.add_argument('--perplexity', type=int, default=30,
						help="Perplexity parameter for t-SNE, consider values between 5 and 50")
	parser.add_argument('--all_embeddings', action="store_true",
						help="visualise both train and test embedding .npz file")
	args = parser.parse_args()

	# Let's plot!
	if args.all_embeddings:
		plotAllEmbeddings(args)
	else:
		plotEmbeddings(args)
	