# Core libraries
import os
import sys
import cv2
import json
import random
import numpy as np

# PyTorch
import torch
from torch.utils import data

# Local libraries
from utilities.ioutils import *

class Pipeline(data.Dataset):
	# Class constructor
	def __init__(	self,
					data_path,
					transform=False,
					img_size=(224, 224),
					suppress_info=True	):
		"""
		Class attributes
		"""

		# The root directory for the dataset itself
		self.__root = data_path
	
		# Whether to transform images/labels into pyTorch form
		self.__transform = transform

		# The image size to resize to
		self.__img_size = img_size

		self.__data = []
		self.__labels = []
		
		"""
		Class setup
		"""

		
		valid_images = ".png"
		for f in os.listdir(self.__root):
			ext = os.path.splitext(f)[1]
			if ext.lower() != valid_images:
				continue
			self.__data.append(os.path.join(self.__root,f))
			self.__labels.append(f)
			# Report some things
		if not suppress_info: self.printStats()


	"""
	Superclass overriding methods
	"""

	# Get the number of items for this dataset (depending on the split)
	def __len__(self):
		return len(self.__data)

	# Index retrieval method
	def __getitem__(self, index):
		# Get and load the anchor image
		img_path = self.__data[index]
		label = self.__labels[index]
		# Load the anchor image
		img_anchor = loadResizeImage(img_path, self.__img_size)
		

		# For sanity checking, visualise the triplet
		# self.__visualiseTriplet(img_anchor, img_pos, img_neg, label_anchor)

		# Transform to pyTorch form
		if self.__transform:
			img_anchor = self.__transformImages(img_anchor)
			

		return img_anchor,label

	"""
	Public methods
	"""	

	# Print stats about the current state of this dataset
	def printStats(self):
		print("Loaded the Data_____________________________")
		print(f"Fold = {int(self.__fold)+1}, split = {self.__split}, combine = {self.__combine}, known = {self.__known}")
		print(f"Found {self.__num_classes} categories: {len(self.__folds_dict[self.__fold]['known'])} known, {len(self.__folds_dict[self.__fold]['unknown'])} unknown")
		print(f"With {len(self.__files['train'])} train images, {len(self.__files['test'])} test images")
		print(f"Unknown categories {self.__folds_dict[self.__fold]['unknown']}")
		print("_______________________________________________________________")

	"""
	(Effectively) private methods
	"""


	# Transform the numpy images into pyTorch form
	def __transformImages(self, img_anchor):
		# Firstly, transform from NHWC -> NCWH
		img_anchor = img_anchor.transpose(2, 0, 1)
		
		# Now convert into pyTorch form
		img_anchor = torch.from_numpy(img_anchor).float()
		
		return img_anchor



# Entry method/unit testing method
if __name__ == '__main__':
	pass