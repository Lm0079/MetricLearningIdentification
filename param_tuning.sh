#!/bin/sh
#SBATCH --job-name dml-param-200
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --time 7-00:00
#SBATCH --mem 64GB

#SBATCH --mail-type=ALL
#SBATCH --mail-user gh18931@bristol.ac.uk

source /user/work/gh18931/diss/MetricLearningIdentification/venv_dml/bin/activate
python param_tuning.py 
deactivate
  