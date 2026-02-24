#!/bin/bash
# Training script for Baseline ViT on CIFAR-10 (without Gram-lowrank)
# For comparison with Gram-ViT

set -e  # Exit on error

# Configuration
WORKDIR="/tmp/baseline_vit_cifar10_$(date +%Y%m%d_%H%M%S)"
MODEL="b16"
DATASET="cifar10"
TOTAL_STEPS=10000
BASE_LR=0.01
BATCH_SIZE=512
EVAL_EVERY=100
CHECKPOINT_EVERY=1000

echo "========================================="
echo "Training Baseline ViT on CIFAR-10"
echo "========================================="
echo "Workdir: $WORKDIR"
echo "Model: ViT-$MODEL (NO Gram-LowRank)"
echo "Total steps: $TOTAL_STEPS"
echo "Batch size: $BATCH_SIZE"
echo "========================================="

# Create workdir if not exists
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
  --config.model.transformer.use_gram_lowrank_mhsa=False

echo "========================================="
echo "Training completed!"
echo "Results saved to: $WORKDIR"
echo "========================================="
