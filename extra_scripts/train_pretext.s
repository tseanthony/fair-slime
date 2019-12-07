#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:p40:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=at4091
#SBATCH --mail-type=END
#SBATCH --mail-user=at4091@nyu.edu
#SBATCH --output=slurm_%j.out


module load ~/anaconda3/4.7.12

conda activate cv

cd $SCRATCH/stl10

python $HOME/ssl/fair-slime/tools/train.py --config_file ${CONFIGFILE} > ${OUTPUTDIR}/${counter}.out