#!/bin/bash
# Ablation Study: Test different ranks for Gram-LowRank
# Compare performance across ranks: 8, 16, 32, 64, 128

set -e  # Exit on error

# Configuration
BASE_WORKDIR="/tmp/gram_vit_ablation_rank"
MODEL="b16"
DATASET="cifar10"
TOTAL_STEPS=10000
BASE_LR=0.01
BATCH_SIZE=512
EVAL_EVERY=100
CHECKPOINT_EVERY=1000

# Ranks to test
RANKS=(8 16 32 64 128)

echo "========================================="
echo "Gram-ViT Rank Ablation Study"
echo "========================================="
echo "Testing ranks: ${RANKS[@]}"
echo "Model: ViT-$MODEL"
echo "Total steps per run: $TOTAL_STEPS"
echo "========================================="

# Loop through each rank
for RANK in "${RANKS[@]}"; do
  WORKDIR="${BASE_WORKDIR}_rank${RANK}_$(date +%Y%m%d_%H%M%S)"

  echo ""
  echo "========================================="
  echo "Training with rank = $RANK"
  echo "Workdir: $WORKDIR"
  echo "========================================="

  # Create workdir
  mkdir -p $WORKDIR

  # Run training
  python -m vit_jax.main \
    --workdir=$WORKDIR \
    --config=vit_jax/configs/vit.py:${MODEL},${DATASET} \
    --config.total_steps=$TOTAL_STEPS \
    --config.base_lr=$BASE_LR \
    --config.batch=$BATCH_SIZE \
    --config.batch_eval=$BATCH_SIZE \
    --config.eval_every=$EVAL_EVERY \
    --config.checkpoint_every=$CHECKPOINT_EVERY \
    --config.model.transformer.use_gram_lowrank_mhsa=True \
    --config.model.transformer.gram_lowrank_rank=$RANK \
    --config.model.transformer.gram_lowrank_a_init_std=1e-2

  echo "Completed rank = $RANK"
  echo ""
done

echo "========================================="
echo "Ablation study completed!"
echo "All results saved to: $BASE_WORKDIR*"
echo "Check W&B for comparison plots"
echo "========================================="
