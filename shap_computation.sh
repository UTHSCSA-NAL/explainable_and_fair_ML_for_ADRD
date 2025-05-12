#!/bin/bash
#SBATCH --job-name=shap_computation
#SBATCH --output=/home/hon3/codes/ad_classification/log/bootstrap_shap_%A_%a.out
#SBATCH --error=/home/hon3/codes/ad_classification/log/bootstrap_shap_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1

PICKLE_FILE="/home/hon3/codes/ad_classification/results/NCvsDem_train-All_rate_0_alpha_0.0_none_0204PM_Mar242025.pkl"
TRAIN_MODE="all"      # or "single"
MODEL_TYPE="xgb"      # change if needed
SCALING_MODE=4
NUM_ITERATION=5000

for (( i=1; i<=NUM_ITERATION; i++ ))
do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=shap_bootstrap_${i}
#SBATCH --output=/home/hon3/codes/ad_classification/log/shap_iter_${i}.out
#SBATCH --error=/home/hon3/codes/ad_classification/log/shap_iter_${i}.err
#SBATCH --time=12:00:00
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Load environment
source /home/hon3/.bashrc
conda activate torch_env

# Run the bootstrap computation script.
python /home/hon3/codes/ad_classification/shap_computation_w_bootstrap.py \
    ${i} ${PICKLE_FILE} ${TRAIN_MODE} ${MODEL_TYPE} ${SCALING_MODE}
EOF
done