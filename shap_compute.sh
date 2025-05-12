#!/bin/bash
# Arrays of input models
# PICKLE_FILES=(
#     "/mnt/d/Projects/ad_classification/results/xgb_new/NCvsDem_train-All_rate_0_alpha_0.0_none_0204PM_Mar242025.pkl"
#     "/mnt/d/Projects/ad_classification/results/xgb_new/NCvsDem_train-NHW_rate_0_alpha_0.0_none_0226PM_Mar242025.pkl"
#     "/mnt/d/Projects/ad_classification/results/xgb_new/NCvsDem_train-NHA_rate_0_alpha_0.0_none_0307PM_Mar242025.pkl"
#     "/mnt/d/Projects/ad_classification/results/xgb_new/NCvsDem_train-Hispanic_rate_0_alpha_0.0_none_0331PM_Mar242025.pkl"
#     "/mnt/d/Projects/ad_classification/results/xgb_new/NCvsDem_train-All_rate_0_alpha_0.0_kmm_0355PM_Mar242025.pkl"
#     "/mnt/d/Projects/ad_classification/results/xgb_new/NCvsDem_train-NHW_rate_0_alpha_0.0_kmm_0406PM_Mar242025.pkl"
#     "/mnt/d/Projects/ad_classification/results/xgb_new/NCvsDem_train-NHA_rate_0_alpha_0.0_kmm_0435PM_Mar242025.pkl"
#     "/mnt/d/Projects/ad_classification/results/xgb_new/NCvsDem_train-Hispanic_rate_0_alpha_0.0_kmm_0454PM_Mar242025.pkl"
#     "/mnt/d/Projects/ad_classification/results/xgb_new/NCvsDem_train-All_rate_0_alpha_0.05_none_cr_1025PM_Mar242025.pkl"
#     "/mnt/d/Projects/ad_classification/results/xgb_new/NCvsDem_train-NHW_rate_0_alpha_0.05_none_cr_1046PM_Mar242025.pkl"
#     "/mnt/d/Projects/ad_classification/results/xgb_new/NCvsDem_train-NHA_rate_0_alpha_0.05_none_cr_1125PM_Mar242025.pkl"
#     "/mnt/d/Projects/ad_classification/results/xgb_new/NCvsDem_train-Hispanic_rate_0_alpha_0.05_none_cr_1148PM_Mar242025.pkl"
#     "/mnt/d/Projects/ad_classification/results/xgb_new/NCvsDem_train-All_rate_0_alpha_0.0_none_harm_1239AM_Mar262025.pkl"
#     "/mnt/d/Projects/ad_classification/results/xgb_new/NCvsDem_train-NHW_rate_0_alpha_0.0_none_harm_0100AM_Mar262025.pkl"
#     "/mnt/d/Projects/ad_classification/results/xgb_new/NCvsDem_train-NHA_rate_0_alpha_0.0_none_harm_0139AM_Mar262025.pkl"
#     "/mnt/d/Projects/ad_classification/results/xgb_new/NCvsDem_train-Hispanic_rate_0_alpha_0.0_none_harm_0201AM_Mar262025.pkl"
# )
# TRAIN_MODES=("all" "nhw_single" "nha_single" "hisp_single" "all" "nhw_single" "nha_single" "hisp_single" "all" "nhw_single" "nha_single" "hisp_single" "all" "nhw_single" "nha_single" "hisp_single")
# MODEL_TYPES=("xgb" "xgb" "xgb" "xgb" "xgb_kmm" "xgb_kmm" "xgb_kmm" "xgb_kmm" "xgb_cr" "xgb_cr" "xgb_cr" "xgb_cr" "xgb_harm" "xgb_harm" "xgb_harm")
# SCALING_MODE=4

# # Loop over each file
# for idx in "${!PICKLE_FILES[@]}"; do
#     PICKLE_FILE="${PICKLE_FILES[$idx]}"
#     TRAIN_MODE="${TRAIN_MODES[$idx]}"
#     MODEL_TYPE="${MODEL_TYPES[$idx]}"

# # Run SHAP computation directly (no bootstrap)
# python3 /mnt/d/Projects/ad_classification/shap_computation_direct.py \
#     ${PICKLE_FILE} ${TRAIN_MODE} ${MODEL_TYPE} ${SCALING_MODE}
# done

SCENARIOS=("all" "nhw" "nha" "hisp")
MODEL_BASE_DIR="/home/Codes/ad_classification/models/xgb_objective_v2_1B"
DATA_DIR="/home/Codes/ad_classification/data/split_fold_w_augmented"
SHAP_SCRIPT="/home/Codes/ad_classification/shap_computation_direct.py"
SCALING_MODE=4
MODEL_TYPE="xgb_regalign"

for SCENARIO in "${SCENARIOS[@]}"; do
    python "$SHAP_SCRIPT" "$SCENARIO" "$DATA_DIR" "$MODEL_BASE_DIR" "$MODEL_TYPE" "$SCALING_MODE"
done
