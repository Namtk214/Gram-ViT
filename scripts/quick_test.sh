#!/bin/bash
# Quick test script to verify setup and Gram-ViT implementation
# Runs a short training session (100 steps) for testing

set -e  # Exit on error

# Configuration
WORKDIR="/tmp/gram_vit_test_$(date +%Y%m%d_%H%M%S)"
MODEL="testing"  # Use small test config
DATASET="cifar10"
TOTAL_STEPS=100
BASE_LR=0.01
BATCH_SIZE=64
EVAL_EVERY=50

echo "========================================="
echo "Quick Test: Gram-ViT Setup"
echo "========================================="
echo "This is a quick test run (100 steps)"
echo "Workdir: $WORKDIR"
echo "========================================="

# Create workdir
mkdir -p $WORKDIR

echo ""
echo "Testing with Gram-LowRank enabled..."
echo ""

# Run quick test
python -m vit_jax.main \
  --workdir=$WORKDIR \
  --config=vit_jax/configs/vit.py:${MODEL},${DATASET} \
  --config.total_steps=$TOTAL_STEPS \
  --config.base_lr=$BASE_LR \
  --config.batch=$BATCH_SIZE \
  --config.batch_eval=$BATCH_SIZE \
  --config.eval_every=$EVAL_EVERY \
  --config.model.transformer.use_gram_lowrank_mhsa=True \
  --config.model.transformer.gram_lowrank_rank=8

echo ""
echo "========================================="
echo "Quick test completed successfully!"
echo "========================================="
echo "Setup is working correctly."
echo "You can now run full training with:"
echo "  ./scripts/train_gram_vit.sh"
echo "========================================="
