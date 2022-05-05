from itertools import product
from os import system
from sys import exit
import os


l_r = [0.1,0.5,0.01,0.05]
lambda_r = [0.0001,0.005,0.001,0.05,0.01,0.1,0.5]
epochs = [100,150,200]
k_v = [1,2,3,4,5]
batch_s = [4,8,16,32]
embeddings = [32,64,128]


# This is a list of tuples ordered by args which combo all inputs
arg_combos = product(l_r,lambda_r,epochs,k_v,embeddings,batch_s)
arg_size =  len(list(product(l_r,lambda_r,epochs,k_v,embeddings,batch_s)))
ESTIMATED_TIME_PER_RUN_MINUTES = 180


if arg_size * ESTIMATED_TIME_PER_RUN_MINUTES > (7 * 24 * 60):
  print(f"Too many runs! {arg_size * ESTIMATED_TIME_PER_RUN_MINUTES}")
  exit(1)
print(f" Estimated time:{arg_size * ESTIMATED_TIME_PER_RUN_MINUTES}")
def generate_command(my_lr,  my_lambda,my_epoch,k_v,embedding,batch):
  return f"python3 train.py --folds_file=/user/work/gh18931/diss/datasets/grevvy_dataset/splits/folds.json --batch_size={batch} --embedding_size={embedding}  --dataset=Zebra  --learning_rate={my_lr} --triplet_lambda={my_lambda} --num_epochs={my_epoch} --n_neighbours={k_v}"

def generate_unique_name(my_lr, my_lambda,my_epoch,k_v,embedding,batch):
  # this might be dangerous depending on your types of arguments, e.g. if they aren't supported in filesystem names
  # in this case it's probably fine
  return f"OUTPUT-lr_{my_lr}-lambda_{my_lambda}-epoch_{my_epoch}-k_{k_v}-embedding_{embedding}-batch_{batch}"

# make sure you mkdir "output" beforehand
for combo in arg_combos:
  unique_name = generate_unique_name(*combo)

  print(f"Starting {unique_name}")
  output_path = os.path.join("/user/work/gh18931/diss/MetricLearningIdentification/output/param_tuning",unique_name)
  os.makedirs(output_path, exist_ok=True)
  system(generate_command(*combo) + f" --out_path=output/{unique_name} | tee logs/{unique_name}.log")