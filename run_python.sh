#!/bin/bash
source /home/henry-ho/miniconda3/etc/profile.d/conda.sh
conda activate torch260

# Data scaling value
train_drop_rate=0
train_types=("All" "NHW" "NHA" "Hispanic")
data_scaling=4
sample_weight_method="none"
corr_remove=0
corr_alphas=0
model_name="xgboost"
sim_threshold=0

# Loop through each train type and run the python script with the corresponding arguments
for train_type in "${train_types[@]}"
do
  #for corr_alpha in "${corr_alphas[@]}"
  #do
    #if [ "$train_type" == "All" ] || [ "$train_type" == "NHW" ]; then
    #    sim_threshold=1.2
    #elif [ "$train_type" == "NHA" ] || [ "$train_type" == "Hispanic" ]; then
    #    sim_threshold=1.8
    #fi
    echo "Running with --train_type $train_type | --model_name $model_name | --data_scaling $data_scaling | --sample_weight_method $sample_weight_method | --corr_remove $corr_remove | --corr_alpha $corr_alpha"
    python train_ml_gridsearch_cv.py --train_type "$train_type" --train_drop_rate "$train_drop_rate" --model_name "$model_name" --data_scaling "$data_scaling" --sample_weight_method "$sample_weight_method" --corr_remove "$corr_remove" --corr_alpha "$corr_alphas" --sim_threshold "$sim_threshold"
  #done
done
