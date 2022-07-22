#!/bin/sh
#SBATCH --job-name dml-1
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --time 0-06:00
#SBATCH --mem 64GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user gh18931@bristol.ac.uk

source /user/work/gh18931/diss/MetricLearningIdentification/venv_dml/bin/activate
python train.py --out_path=output/research/tex_80 --eval_freq=2 --folds_file=/user/work/gh18931/diss/datasets/grevvy_dataset/splits/folds.json  --dataset=Zebra --n_neighbours=1 --batch_size=8 --triplet_lambda=0.0001 --learning_rate=0.1 --num_epochs=200 

deactivate
  