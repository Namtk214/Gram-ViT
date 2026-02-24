# Gram-ViT Training Guide

Hướng dẫn train Vision Transformer với Gram + Low-Rank Residual trên CIFAR-10 sử dụng JAX/Flax.

---

## 📋 Tổng quan

Project này implement **Gram + Low-Rank Residual** cho MHSA output trong Vision Transformer (ViT). Correction term được thêm vào attention branch theo công thức:

```
Y = Z + T
T = G_t(AB)
G_t = XX^T / D (token Gram matrix)
```

với:
- A: matrix [N, r] (khởi tạo Normal(0, 0.01))
- B: matrix [r, D] (khởi tạo zeros - no-op ban đầu)
- r = 64 (rank)

---

## 🛠️ Yêu cầu hệ thống

### Hardware
- **GPU**: NVIDIA GPU với CUDA support (khuyến nghị: ≥8GB VRAM)
- **RAM**: ≥16GB
- **Storage**: ≥10GB free space (cho dataset và checkpoints)

### Software
- Python 3.8+
- CUDA 11.x
- cuDNN 8.6+

---

## 📦 Cài đặt

### 1. Clone repository
```bash
cd /Users/ngothanhnam/Desktop/Gram-ViT/Gram-ViT
```

### 2. Tạo virtual environment (khuyến nghị)
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# hoặc
venv\Scripts\activate  # Windows
```

### 3. Cài đặt dependencies
```bash
cd vit_jax
pip install -r requirements.txt
```

**Lưu ý**: File requirements.txt bao gồm JAX với CUDA support. Nếu bạn chỉ có CPU:
```bash
pip install jax[cpu]
```

### 4. Cài đặt Weights & Biases
```bash
pip install wandb
wandb login
```

Nhập API key của bạn khi được yêu cầu (lấy từ https://wandb.ai/authorize)

---

## 🚀 Chạy Training

### Basic Training Command

```bash
python -m vit_jax.main \
  --workdir=/tmp/gram_vit_cifar10 \
  --config=vit_jax/configs/vit.py:b16,cifar10
```

### Giải thích các tham số:

- `--workdir`: Thư mục lưu checkpoints, logs, và W&B data
- `--config`: Config file với format `<file>:<model>,<dataset>`
  - `vit.py`: Config file
  - `b16`: ViT-Base/16 model
  - `cifar10`: CIFAR-10 dataset

### Các model variants khả dụng:

| Model | Description | Parameters |
|-------|-------------|------------|
| `ti16` | ViT-Tiny/16 | 5.7M |
| `s16` | ViT-Small/16 | 22M |
| `b16` | ViT-Base/16 | 86M |
| `l16` | ViT-Large/16 | 307M |
| `b32` | ViT-Base/32 | 88M |

**Ví dụ với ViT-Small:**
```bash
python -m vit_jax.main \
  --workdir=/tmp/gram_vit_small \
  --config=vit_jax/configs/vit.py:s16,cifar10
```

---

## ⚙️ Configuration Options

### Modify Training Hyperparameters

Bạn có thể override config parameters trực tiếp từ command line:

```bash
python -m vit_jax.main \
  --workdir=/tmp/gram_vit_cifar10 \
  --config=vit_jax/configs/vit.py:b16,cifar10 \
  --config.base_lr=0.001 \
  --config.total_steps=20000 \
  --config.batch=256 \
  --config.eval_every=500
```

### Các config quan trọng:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_lr` | 0.01 (b16+cifar10) | Learning rate |
| `total_steps` | 10,000 | Tổng số training steps |
| `batch` | 512 | Batch size cho training |
| `batch_eval` | 512 | Batch size cho evaluation |
| `warmup_steps` | 500 | Warmup steps cho LR scheduler |
| `eval_every` | 100 | Evaluate sau mỗi N steps |
| `progress_every` | 10 | Log progress sau mỗi N steps |
| `checkpoint_every` | 1,000 | Save checkpoint sau mỗi N steps |

### Gram-LowRank specific configs:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.transformer.use_gram_lowrank_mhsa` | True | Enable Gram-lowrank residual |
| `model.transformer.gram_lowrank_rank` | 64 | Rank cho low-rank matrices |
| `model.transformer.gram_lowrank_a_init_std` | 1e-2 | Std dev cho init A matrix |

**Ví dụ tắt Gram-lowrank (baseline ViT):**
```bash
python -m vit_jax.main \
  --workdir=/tmp/baseline_vit \
  --config=vit_jax/configs/vit.py:b16,cifar10 \
  --config.model.transformer.use_gram_lowrank_mhsa=False
```

**Ví dụ thay đổi rank:**
```bash
python -m vit_jax.main \
  --workdir=/tmp/gram_vit_rank128 \
  --config=vit_jax/configs/vit.py:b16,cifar10 \
  --config.model.transformer.gram_lowrank_rank=128
```

---

## 📊 Weights & Biases Logging

### Metrics được log tự động:

#### Priority 1 - Core Metrics
- **Training**: `Train/loss`, `Train/learning_rate`
- **Validation**: `Val/loss`, `Val/accuracy`, `Val/top5_accuracy`
- **Per-class accuracy**: `Val/per_class_accuracy/{class_name}` (10 classes)
- **Confusion Matrix**: `Charts/confusion_matrix`

#### Priority 2 - Optimization
- **Gradients**: `Optim/grad_global_norm`
- **Parameters**: `Optim/param_global_norm`
- **Activations**: `Activations/block_i/mhsa_out_mean`, `Activations/block_i/mhsa_out_std`
- **MLP**: `Activations/block_i/mlp_out_mean`, `Activations/block_i/mlp_out_std`

#### Gram-LowRank Specific Metrics
Cho mỗi encoder block:
- `GramLowRank/block_i/T_norm` - Norm của correction term
- `GramLowRank/block_i/Z_norm` - Norm của MHSA output
- `GramLowRank/block_i/T_over_Z_norm` - Tỷ lệ T/Z (quan trọng!)
- `GramLowRank/block_i/A_norm` - Norm của matrix A
- `GramLowRank/block_i/B_norm` - Norm của matrix B

### Xem results trên W&B:

1. Truy cập: https://wandb.ai/
2. Project name: `gram-vit-cifar10`
3. Run name: `{model_name}_{dataset}`

---

## 💾 Checkpointing

### Automatic Checkpointing

Checkpoints được tự động lưu mỗi `checkpoint_every` steps tại `workdir/`.

### Resume Training từ Checkpoint

```bash
python -m vit_jax.main \
  --workdir=/tmp/gram_vit_cifar10 \
  --config=vit_jax/configs/vit.py:b16,cifar10
```

Training sẽ tự động resume từ checkpoint mới nhất trong workdir nếu có.

### Checkpoint Format

```
workdir/
├── checkpoint_1000
├── checkpoint_2000
├── checkpoint_3000
└── ...
```

Mỗi checkpoint chứa:
- Model parameters
- Optimizer state
- Current step

---

## 🎯 Training from Scratch vs Fine-tuning

### Training from Scratch (Recommended cho CIFAR-10)

```bash
python -m vit_jax.main \
  --workdir=/tmp/gram_vit_scratch \
  --config=vit_jax/configs/vit.py:b16,cifar10 \
  --config.pretrained_dir='.'
```

**Lưu ý**: Code hiện tại yêu cầu pretrained checkpoint. Để train from scratch hoàn toàn, cần sửa code để skip checkpoint loading.

### Alternative: Quick Testing Config

Sử dụng testing config cho debugging nhanh:
```bash
python -m vit_jax.main \
  --workdir=/tmp/test_run \
  --config=vit_jax/configs/vit.py:testing,cifar10 \
  --config.total_steps=100 \
  --config.eval_every=50
```

---

## 📈 Monitoring Training

### Real-time Logs

Training progress được log ra console:
```
Step: 100/10000 1.0%, img/sec/core: 234.5, ETA: 2.45h
Step: 100 Learning rate: 0.0010000, Test accuracy: 0.45123, img/sec/core: 456.7
```

### W&B Dashboard

Xem real-time metrics:
- Learning curves
- Confusion matrix
- Gradient/parameter norms
- Gram-lowrank stats (T/Z ratio)

### Check GPU Usage

```bash
nvidia-smi -l 1  # Update mỗi giây
```

---

## 🐛 Troubleshooting

### 1. Out of Memory (OOM)

**Giảm batch size:**
```bash
--config.batch=256 --config.batch_eval=256
```

**Tăng gradient accumulation:**
```bash
--config.accum_steps=16
```

### 2. Pretrained Checkpoint Not Found

Error: `Could not find "path/to/model.npz"`

**Solution**: Download pretrained model hoặc train from scratch:
```bash
# Option 1: Download từ Google Cloud
gsutil -m cp gs://vit_models/imagenet21k/ViT-B_16.npz /path/to/pretrained/

# Option 2: Sửa config để skip pretrained (cần modify code)
```

### 3. W&B Login Issues

```bash
wandb login --relogin
```

Hoặc offline mode:
```bash
wandb offline
```

### 4. JAX/CUDA Version Mismatch

Reinstall JAX với đúng CUDA version:
```bash
pip uninstall jax jaxlib
pip install --upgrade "jax[cuda11_cudnn86]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 5. Slow Training / Low GPU Utilization

- Check data pipeline bottleneck: tăng `--config.prefetch=4`
- Verify XLA compilation: logs sẽ show "Compiling..."
- Monitor with `nvidia-smi` và `htop`

---

## 📝 Example Training Scripts

### Full Training Run (ViT-B/16 with Gram-lowrank)

```bash
#!/bin/bash

python -m vit_jax.main \
  --workdir=/tmp/gram_vit_full \
  --config=vit_jax/configs/vit.py:b16,cifar10 \
  --config.total_steps=10000 \
  --config.base_lr=0.01 \
  --config.batch=512 \
  --config.eval_every=100 \
  --config.checkpoint_every=1000 \
  --config.model.transformer.use_gram_lowrank_mhsa=True \
  --config.model.transformer.gram_lowrank_rank=64
```

### Baseline ViT (No Gram-lowrank)

```bash
#!/bin/bash

python -m vit_jax.main \
  --workdir=/tmp/baseline_vit \
  --config=vit_jax/configs/vit.py:b16,cifar10 \
  --config.total_steps=10000 \
  --config.base_lr=0.01 \
  --config.batch=512 \
  --config.eval_every=100 \
  --config.model.transformer.use_gram_lowrank_mhsa=False
```

### Ablation Study: Different Ranks

```bash
#!/bin/bash

for rank in 8 16 32 64 128; do
  python -m vit_jax.main \
    --workdir=/tmp/gram_vit_rank${rank} \
    --config=vit_jax/configs/vit.py:b16,cifar10 \
    --config.model.transformer.gram_lowrank_rank=${rank}
done
```

---

## 📚 Reference

### Config Files
- `vit_jax/configs/vit.py` - Main ViT configs
- `vit_jax/configs/common.py` - Common training configs
- `vit_jax/configs/models.py` - Model architecture configs

### Key Files
- `vit_jax/main.py` - Entry point
- `vit_jax/train.py` - Training loop với W&B logging
- `vit_jax/models_vit.py` - ViT architecture với Gram-lowrank
- `vit_jax/input_pipeline.py` - Data loading

### Important Papers
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) - Original ViT paper
- [How to train your ViT?](https://arxiv.org/abs/2106.10270) - Training best practices

---

## 🎓 Tips for Success

1. **Start Small**: Test với `testing` config hoặc `s16` model trước
2. **Monitor W&B**: Theo dõi `T_over_Z_norm` để xem Gram-lowrank có hoạt động
3. **Baseline First**: Train baseline ViT để so sánh
4. **Ablation Studies**: Test different ranks (8, 16, 32, 64, 128)
5. **GPU Warm-up**: First step sẽ compile chậm (1-2 phút), bình thường!

---

## ❓ FAQ

**Q: Training mất bao lâu?**
A: ViT-B/16 trên CIFAR-10, 10K steps: ~2-3 giờ trên V100 GPU

**Q: Làm sao biết Gram-lowrank có hiệu quả?**
A: Xem metric `T_over_Z_norm` trên W&B. Nếu > 0 và tăng dần, branch đang học.

**Q: Tại sao B matrix init = 0?**
A: LoRA-style initialization. Đảm bảo model bắt đầu giống baseline ViT.

**Q: Có thể train trên CPU không?**
A: Có nhưng rất chậm (>100x). Không khuyến nghị.

**Q: Dataset được download tự động?**
A: Có, TensorFlow Datasets sẽ tự download CIFAR-10 lần đầu chạy.

---

## 📞 Support

Nếu gặp vấn đề:
1. Check Troubleshooting section
2. Check W&B logs để debug
3. Verify JAX/GPU setup: `python -c "import jax; print(jax.devices())"`

Happy Training! 🚀
