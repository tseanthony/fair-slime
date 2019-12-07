# Loop for calling sbatch script

#!/bin/bash

# directory containing YAML files
YAMLDIR="$HOME/ssl/fair-slime/configs/alexnet_jigsaw"

# base name for YAML file
BASENAME="unsupervised_alexnet_jigsaw_stl10_"

# output directory
OUTPUTDIR="$SCRATCH/stl10/output1"

# set number YAML files in loop

counter=1
while [ $counter -le 6 ]
do
CONFIGFILE="${YAMLDIR}/${BASENAME}${counter}.yaml"
export counter
export CONFIGFILE
export OUTPUTDIR
sbatch train_pretext.s
((counter++))
done

