#!/bin/bash
# Compare Gram-ViT vs Baseline ViT
# Runs both variants sequentially for fair comparison

set -e  # Exit on error

# Configuration
BASE_WORKDIR="/tmp/comparison_$(date +%Y%m%d_%H%M%S)"
MODEL="b16"
DATASET="cifar10"
TOTAL_STEPS=10000
BASE_LR=0.01
BATCH_SIZE=512
EVAL_EVERY=100
CHECKPOINT_EVERY=1000

echo "========================================="
echo "Gram-ViT vs Baseline ViT Comparison"
echo "========================================="
echo "Model: ViT-$MODEL"
echo "Steps: $TOTAL_STEPS"
echo "This will run 2 training sessions:"
echo "  1. Baseline ViT (no Gram-lowrank)"
echo "  2. Gram-ViT (rank=64)"
echo "========================================="

# -------------------------------------------
# Run 1: Baseline ViT
# -------------------------------------------
echo ""
echo "========================================="
echo "RUN 1: Baseline ViT"
echo "========================================="

WORKDIR_BASELINE="${BASE_WORKDIR}/baseline"
mkdir -p $WORKDIR_BASELINE

python -m vit_jax.main \
  --workdir=$WORKDIR_BASELINE \
  --config=vit_jax/configs/vit.py:${MODEL},${DATASET} \
  --config.total_steps=$TOTAL_STEPS \
  --config.base_lr=$BASE_LR \
  --config.batch=$BATCH_SIZE \
  --config.batch_eval=$BATCH_SIZE \
  --config.eval_every=$EVAL_EVERY \
  --config.checkpoint_every=$CHECKPOINT_EVERY \
  --config.model.transformer.use_gram_lowrank_mhsa=False

echo "Baseline ViT completed!"

# -------------------------------------------
# Run 2: Gram-ViT
# -------------------------------------------
echo ""
echo "========================================="
echo "RUN 2: Gram-ViT (rank=64)"
echo "========================================="

WORKDIR_GRAM="${BASE_WORKDIR}/gram_rank64"
mkdir -p $WORKDIR_GRAM

python -m vit_jax.main \
  --workdir=$WORKDIR_GRAM \
  --config=vit_jax/configs/vit.py:${MODEL},${DATASET} \
  --config.total_steps=$TOTAL_STEPS \
  --config.base_lr=$BASE_LR \
  --config.batch=$BATCH_SIZE \
  --config.batch_eval=$BATCH_SIZE \
  --config.eval_every=$EVAL_EVERY \
  --config.checkpoint_every=$CHECKPOINT_EVERY \
  --config.model.transformer.use_gram_lowrank_mhsa=True \
  --config.model.transformer.gram_lowrank_rank=64 \
  --config.model.transformer.gram_lowrank_a_init_std=1e-2

echo "Gram-ViT completed!"

# -------------------------------------------
# Summary
# -------------------------------------------
echo ""
echo "========================================="
echo "Comparison completed!"
echo "========================================="
echo "Results:"
echo "  Baseline: $WORKDIR_BASELINE"
echo "  Gram-ViT: $WORKDIR_GRAM"
echo ""
echo "Compare on W&B:"
echo "  Project: gram-vit-cifar10"
echo "  Look for runs from today"
echo "========================================="
