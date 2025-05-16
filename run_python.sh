#!/bin/bash

# Data scaling value
train_drop_rate=0
train_types=("All" "NHW" "NHA" "Hispanic")
data_scaling=4
sample_weight_method="none"
corr_remove=1
corr_alphas=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0)
model_name="xgboost"
sim_threshold=0

# Loop through each train type and run the python script with the corresponding arguments
for train_type in "${train_types[@]}"
do
  for corr_alpha in "${corr_alphas[@]}"
  do
    #if [ "$train_type" == "All" ] || [ "$train_type" == "NHW" ]; then
    #    sim_threshold=1.2
    #elif [ "$train_type" == "NHA" ] || [ "$train_type" == "Hispanic" ]; then
    #    sim_threshold=1.8
    #fi
    echo "Running with --train_type $train_type | --model_name $model_name | --data_scaling $data_scaling | --sample_weight_method $sample_weight_method"
    echo "             --corr_remove $corr_remove | --corr_alpha $corr_alpha"
    python train_ml_gridsearch_cv.py --train_type "$train_type" --train_drop_rate "$train_drop_rate" --model_name "$model_name" --data_scaling "$data_scaling" --sample_weight_method "$sample_weight_method" --corr_remove "$corr_remove" --corr_alpha "$corr_alpha" --sim_threshold "$sim_threshold"
  done
done
