#!/bin/bash

# ---------- CONFIGURATION ----------
mode="$1"  # Either "mode1" or "mode2"
echo "Selected mode: $mode"

# ---------- MODE 1: train_ml_gridsearch_cv.py ----------
if [[ "$mode" == "mode1" ]]; then
    train_drop_rate=0
    train_types=("All")  # Can be expanded to "NHW" "NHA" "Hispanic"
    data_scaling=4
    sample_weight_method="none"
    corr_remove=0
    corr_alpha=0.05
    model_name="xgboost"
    sim_threshold=0
    n_splits=10
    harmonize=0

    # Customizable paths
    df_roi="./data/filtered_data_09182024.csv" # Use filtered_data_09182024_Harmonized_DX.csv for harmonize=1
    save_model_path="./models/xgb"
    save_result_path="./results/xgb"

    for train_type in "${train_types[@]}"
    do
        echo "Running train_ml_gridsearch_cv.py with:"
        echo "  --train_type $train_type"
        echo "  --model_name $model_name"
        echo "  --data_scaling $data_scaling"
        echo "  --sample_weight_method $sample_weight_method"
        echo "  --corr_remove $corr_remove"
        echo "  --corr_alpha $corr_alpha"
        echo "  --n_splits $n_splits"
        echo "  --harmonize $harmonize"
        echo "  --save_model_path $save_model_path"
        echo "  --save_result_path $save_result_path"
        echo "  --df_roi $df_roi"

        python train_ml_gridsearch_cv.py \
            --train_type "$train_type" \
            --train_drop_rate "$train_drop_rate" \
            --model_name "$model_name" \
            --sample_weight_method "$sample_weight_method" \
            --n_splits "$n_splits" \
            --data_scaling "$data_scaling" \
            --corr_remove "$corr_remove" \
            --corr_alpha "$corr_alpha" \
            --harmonize "$harmonize" \
            --sim_threshold "$sim_threshold" \
            --df_roi "$df_roi" \
            --save_model_path "$save_model_path" \
            --save_result_path "$save_result_path"
    done

# ---------- MODE 2: train_custom_obj.py ----------
elif [[ "$mode" == "mode2" ]]; then
    # Configure scenario-level arguments
    scenarios=("All")
    objective_subtype="1C"  # or "2B"
    save_model_path="./models"
    save_result_path="./results"
    num_few_shot=5
    total_steps=1000
    inner_splits=10
    outer_splits=10
    n_job=8

    for scenario in "${scenarios[@]}"
    do
        echo "Running train_custom_obj.py with:"
        echo "  --scenario $scenario"
        echo "  --objective_subtype $objective_subtype"
        echo "  --few-shot num: $num_few_shot | total_steps: $total_steps"

        python train_custom_obj.py \
            --scenario "$scenario" \
            --save_model_path "$save_model_path" \
            --save_result_path "$save_result_path" \
            --objective_subtype "$objective_subtype" \
            --num_few_shot "$num_few_shot" \
            --total_steps "$total_steps" \
            --inner_splits "$inner_splits" \
            --outer_splits "$outer_splits" \
            --n_job "$n_job"
    done

else
    echo "❌ Error: Unknown mode '$mode'. Please use 'mode_1' or 'mode_2'."
    exit 1
fi

echo "✅ All jobs completed."