#!/usr/bin/env python3

from ast import Continue
import subprocess

import argparse

import os
import csv
parser = argparse.ArgumentParser(
    description=" Visualising DML through PCA",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

path = '/user/work/gh18931/diss/MetricLearningIdentification/output/param_tune'
folds_file = '/user/work/gh18931/diss/datasets/grevvy_dataset/splits/folds.json'
def test( k_value, checkpoint_path,distance,batch_v,embedding):
        # Construct subprocess call string
        run_str  = f"python test.py"
        run_str += f" --model_path={checkpoint_path}"  # Saved model weights to use
        run_str += f" --dataset=Zebra"        # Which dataset to use
        run_str += f" --batch_size={batch_v}"  # Batch size to use when inferring
        run_str += f" --embedding_size={embedding}"  # Embedding dimensionality
        run_str += f" --current_fold=0"  # The current fold number
        run_str += f" --folds_file={folds_file}"  # Where to find info about this fold
        run_str += f" --save_path=./"    # Where to store the embeddings
        run_str += f" --n_neighbours={k_value}"  
        run_str += f" --save_embeddings=False"
        run_str += f" --distance_metric={distance}"

        # Let's run the command, decode and save the result
        
        try:
        
            accuracy = subprocess.check_output([run_str], shell=True)
            accuracy = float(accuracy.decode('utf-8').split("Accuracy=")[1])
        except subprocess.CalledProcessError as e:
            accuracy = -1.0
    

        # Report the accuracy
        print(f"Accuracy: {accuracy}%")
        return accuracy
def get_data(distance_metric):
    i = 0
    accuracies = {}
    for  data_path in [path]:
        valid_images = ".log"
      
        for f in os.listdir(data_path):
            if not "OUTPUT-lr" in f :
                continue
            print(f)
            i +=1
            print(i)
            checkpoint_path = os.path.join(data_path,f,"fold_0","best_model_state.pkl")
            if os.path.isfile(checkpoint_path):
                k_value = str(f).split("k_")[-1].split("-")[0]
                if "batch" in f:
                    batch_v = str(f).split("batch_")[-1].split(".log")[0]
                else:
                    batch_v = 16
                if "embedding" in f:

                    embedding_v = str(f).split("embedding_")[-1].split("-")[0]
                else: 
                    embedding_v = 128
                accuracies[f]= test(k_value,checkpoint_path,distance_metric,batch_v,embedding_v)
    
    return accuracies
def output(filename,my_dict):
    with open(filename, 'w') as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, my_dict.keys())
        w.writeheader()
        w.writerow(my_dict)
def collection():
    distance = get_data(True)
    output("data_hyperparameter_tuning_FINAL_DATASET.csv",distance)
def read_file(name):
    
    dict_from_csv = {}

    with open(name , mode='r') as inp:
        reader = csv.reader(inp)
        reader = list(reader)
        
        for i in range(len(reader[0])):
            dict_from_csv[reader[0][i]]= float(reader[1][i])

        

    
    return dict_from_csv
def visualise():
    distance = read_file("data_hyperparameter_tuning_FINAL_DATASET.csv")
    for acc in distance:
        if distance[acc] >51:
         print(f" {acc}: accuracy={distance[acc]}")
    

    

def main(args):
    #collection()
    visualise()
   
if __name__ == "__main__":
    main(parser.parse_args())
