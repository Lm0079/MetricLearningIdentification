#!/usr/bin/env python3
import argparse
import multiprocessing
from pathlib import Path
import numpy as np
import os
import matplotlib
import subprocess
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
	description=" DML Learning curves",
	formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
	"--out",
	type=Path,
	help="Filepath for out file for the run "
)

def get_data(args):
	with open(args.out) as f:
		lines = f.readlines()
		loss = []
		accuracy = []
		for line in lines:
			if "loss_mean" in line:
				
				loss.append(float(line.split("loss_mean:")[-1].split(",")[0]))
			if "Accuracy:" in line:
				
				accuracy.append(float(line.split(":")[-1].split("%")[0]))
		fig , ax = plt.subplots()
		
		diff = int(len(loss)/len(accuracy))

		x_loss = np.arange(0, len(loss))
		ax.plot(x_loss,loss,color="g")
		ax2 = ax.twinx()
		
		x_acc = np.arange(0,len(accuracy)*diff,diff)
		ax2.plot(x_acc,accuracy)		
		ax.set_ylim([0, 0.1])
		plt.savefig("learning_c.png")
		


def main(args):
	get_data(args)
	
   
if __name__ == "__main__":
	main(parser.parse_args())