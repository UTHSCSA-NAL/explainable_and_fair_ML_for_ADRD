#!/bin/bash

# Data scaling value
train_drop_rate=0
train_types=("NHW" "NHA")
data_scaling=4
sample_weight_method="none"
corr_remove=0
corr_alpha=0.05
model_name="xgboost"
sim_threshold=0

# Loop through each train type and run the python script with the corresponding arguments
for train_type in "${train_types[@]}"
do
    echo "Running with --train_type $train_type | --model_name $model_name | --data_scaling $data_scaling | --sample_weight_method $sample_weight_method"
    echo "             --corr_remove $corr_remove | --corr_alpha $corr_alpha"
    python train_ml_gridsearch_cv.py --train_type "$train_type" --train_drop_rate "$train_drop_rate" --model_name "$model_name" --data_scaling "$data_scaling" --sample_weight_method "$sample_weight_method" --corr_remove "$corr_remove" --corr_alpha "$corr_alpha" --sim_threshold "$sim_threshold"
done
echo "All training completed."
# End of script