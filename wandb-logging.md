# W&B Logging Plan for ViT CIFAR-10 (Flax/JAX)

## 1) Mục tiêu

Thiết kế hệ thống logging trên Weights & Biases (W&B) để:
- Theo dõi quá trình học (learning curves)
- Kiểm soát tối ưu (LR, gradients, stability)
- Phân tích hành vi ViT (attention maps / rollout / activation)
- Đánh giá sâu (confusion matrix, per-class accuracy)
- Theo dõi tài nguyên (GPU/CPU/RAM - W&B thường tự log)

Áp dụng cho codebase ViT hiện tại dùng:
- Flax / JAX
- Optax
- CIFAR-10 classifier

---

## 2) Phân loại các nhóm metric cần log

## A. Learning Curves cơ bản (BẮT BUỘC)
Các chỉ số để phát hiện underfitting / overfitting:

### Train
- `Train/loss`
- `Train/accuracy` (nếu tính trong train loop)
- `Train/learning_rate`

### Validation
- `Val/loss`
- `Val/accuracy`
- `Val/top5_accuracy` (tùy CIFAR-10 có thể vẫn log để chuẩn hóa pipeline)
- `Val/per_class_accuracy/*` (10 lớp CIFAR-10)

---

## B. Kiểm soát quá trình tối ưu (Optimization)
Giúp phát hiện exploding/vanishing gradients và vấn đề scheduler.

### Scalars
- `Optim/lr`
- `Optim/grad_global_norm`
- `Optim/update_global_norm` (nếu dễ lấy)
- `Optim/param_global_norm`

### Histograms (định kỳ, không log mỗi step)
- `Gradients/<param_path>` histogram
- `Params/<param_path>` histogram
- (tùy chọn) `Updates/<param_path>` histogram

Lưu ý:
- Với Flax/JAX không dùng `wandb.watch(model)` như PyTorch; cần tự chuyển tree params/grads sang numpy và log histogram thủ công.
- Nên log theo tần suất thấp (vd mỗi 100/500 step) để tránh overhead.

---

## C. Activation / Internal States (đặc biệt quan trọng cho ViT)
Mục tiêu:
- Kiểm tra saturation, collapse, dead activations
- Theo dõi phân phối đầu ra MHSA / MLP

### Nên log
- `Activations/encoderblock_i/mhsa_out_mean`
- `Activations/encoderblock_i/mhsa_out_std`
- `Activations/encoderblock_i/mlp_out_mean`
- `Activations/encoderblock_i/mlp_out_std`

### Histograms (định kỳ)
- `ActivationsHist/encoderblock_i/mhsa_out`
- `ActivationsHist/encoderblock_i/mlp_out`

### Với branch Gram-lowrank (nếu đã thêm)
- `GramLowRank/block_i/T_norm`
- `GramLowRank/block_i/Z_norm`
- `GramLowRank/block_i/T_over_Z_norm`
- `GramLowRank/block_i/A_norm`
- `GramLowRank/block_i/B_norm`

---

## D. Phân tích trực quan (Interpretability / Visualization)
## D1. Attention Maps (Rất quan trọng với ViT)
Mục tiêu:
- Xem model chú ý patch nào khi phân loại

### Nên log dạng image/heatmap
- `Vis/attention_map_samples`
- `Vis/attention_cls_to_patch_layer{L}_head{H}`

Cách hiển thị:
- Chọn 4–8 mẫu ảnh cố định (fixed validation samples)
- Với mỗi mẫu, tạo heatmap attention (ví dụ CLS -> patches)
- Log bằng `wandb.Image(...)`

**Lưu ý triển khai với Flax:**
- `flax.linen.MultiHeadDotProductAttention` mặc định không tự trả attention weights ra ngoài trong code hiện tại.
- Cần sửa module attention hoặc block để `sow()` / capture weights trung gian, hoặc tự implement MHSA wrapper trả thêm attention logits/weights.

## D2. Attention Rollout (nâng cao)
Mục tiêu:
- Tổng hợp attention qua nhiều layer/head để thấy dòng chảy thông tin toàn cục

Nên log:
- `Vis/attention_rollout_samples`

Tần suất:
- Mỗi epoch hoặc mỗi `eval_every` (không cần mỗi step)

## D3. Saliency Maps (tùy chọn, hữu ích)
Mục tiêu:
- Xem vùng pixel ảnh hưởng mạnh tới prediction

Gợi ý:
- Tính gradient của logit lớp dự đoán theo input image (JAX grad wrt input)
- Tạo heatmap |grad| hoặc grad * input
- Log bằng `wandb.Image(...)`

Nên log:
- `Vis/saliency_maps`

---

## E. Đánh giá chuyên sâu (Advanced Evaluation)
## E1. Confusion Matrix (BẮT BUỘC cho classifier)
W&B hỗ trợ trực tiếp custom chart confusion matrix bằng `wandb.plot.confusion_matrix(...)`. :contentReference[oaicite:1]{index=1}

Nên log:
- `Charts/confusion_matrix`

Dữ liệu cần:
- `y_true` (label thật)
- `preds` hoặc `probs`
- `class_names` (10 lớp CIFAR-10)

## E2. Per-class Accuracy
Nên log:
- `Val/per_class_accuracy/airplane`
- `Val/per_class_accuracy/automobile`
- ...
- `Val/per_class_accuracy/truck`

## E3. Top-k Accuracy
- `Val/top1_accuracy`
- `Val/top5_accuracy`
(Top-5 vẫn có ý nghĩa để pipeline tái sử dụng cho CIFAR-100/ImageNet)

---

## F. Tài nguyên hệ thống (System)
W&B thường tự log:
- GPU utilization
- GPU memory
- CPU / RAM
- Disk / network (tùy runtime)

Nên theo dõi để phát hiện:
- memory leak
- throughput giảm dần bất thường

---


---

## 4) Tần suất logging đề xuất (để tránh overhead)

## Mỗi step (hoặc mỗi `progress_every`)
Log scalar nhẹ:
- `Train/loss`
- `Optim/lr`
- `Optim/grad_global_norm` (nếu có)

## Mỗi eval (`eval_every`)
Log:
- `Val/loss`
- `Val/accuracy`
- `Val/top5_accuracy`
- `Val/per_class_accuracy/*`
- `Charts/confusion_matrix`

## Mỗi epoch hoặc mỗi vài eval
Log visualization nặng:
- attention maps
- attention rollout
- saliency maps
- activation histograms
- gradient/parameter histograms

Khuyến nghị:
- Không log histogram/heatmap ở mọi step
- Dùng fixed validation samples để so sánh theo thời gian

---

## 5) Cần thay đổi gì trong code hiện tại (mức kiến trúc)

## 5.1 Trong training loop (`train_and_evaluate`)
Hiện code đã log:
- `train_loss`
- `accuracy_test`
- `lr`

Cần bổ sung:
- train accuracy (nếu muốn learning curves đầy đủ)
- val loss
- top-k accuracy
- confusion matrix
- per-class accuracy
- gradient norms
- (tùy chọn) histogram params/grads

### Gợi ý thay đổi nhỏ
- Tạo hàm helper `compute_eval_metrics(...)`
- Tạo hàm helper `log_wandb_metrics(...)`
- Tạo hàm helper `log_wandb_visuals(...)`

---

## 5.2 Trong model / apply_fn để lấy activation và attention
### Activation (Flax)
Dùng một trong các cách:
1. `sow()` trong các module (`Encoder1DBlock`) để ghi intermediate vào collection (ví dụ `"intermediates"`)
2. `capture_intermediates=True` trong `model.apply(...)` (nếu phù hợp)
3. Wrapper trả thêm auxiliary outputs (attention/activations)

### Attention weights
Do `nn.MultiHeadDotProductAttention` hiện đang dùng trực tiếp và không trả weights:
- cần custom wrapper để expose attention weights
- hoặc sửa block để lưu weights trong collection

---

## 5.3 Với JAX/Flax: logging gradient histograms thủ công
Không dùng `wandb.watch(model, log="all")` theo kiểu PyTorch.

Thay vào đó:
- Sau khi có `grads` trong `update_fn`
- Chọn một subset params/grads (vd embedding, head, 1-2 block đầu/cuối)
- Chuyển về numpy (`np.array(...)`)
- Log histogram qua `wandb.Histogram(...)`

Lý do:
- log full tree mỗi step rất nặng và chậm

---

## 6) Danh sách metric chi tiết khuyến nghị cho ViT CIFAR-10

## Core metrics (minimum)
- `Train/loss`
- `Optim/lr`
- `Val/accuracy`
- `Val/loss`
- `Charts/confusion_matrix`

## Strong baseline logging (recommended)
Thêm:
- `Train/accuracy`
- `Val/top5_accuracy`
- `Val/per_class_accuracy/*`
- `Optim/grad_global_norm`
- `Optim/param_global_norm`

## ViT-specific
- attention maps (1–2 layers, 1–2 heads, fixed samples)
- activation mean/std per block
- (nếu có Gram-lowrank) `T_over_Z_norm`

## Advanced 
- attention rollout
- saliency maps
- histogram params/grads/activations
- position embedding similarity heatmap

---

## 7) Position Embedding Similarity (ViT-specific)
Mục tiêu:
- Kiểm tra quan hệ giữa learned positional embeddings

Cách log:
- Lấy `pos_embedding` từ `AddPositionEmbs`
- Chuẩn hóa vector theo chiều embedding
- Tính cosine similarity matrix giữa các vị trí
- Log heatmap bằng `wandb.Image(...)` (hoặc matplotlib figure convert sang image)

Nên log:
- `Vis/pos_embedding_similarity`

Tần suất:
- mỗi eval lớn hoặc mỗi epoch

---


---

## 9) Mapping trực tiếp từ yêu cầu hiện tại sang implementation priority

### Ưu tiên 1 (triển khai ngay)
- Train Loss
- Val Loss / Val Accuracy
- LR
- Confusion Matrix
- Per-class Accuracy

### Ưu tiên 2 (ổn định hóa mô hình)
- Grad global norm
- Param global norm
- Activation mean/std per block

### Ưu tiên 3 (phân tích ViT)
- Attention Map
- Position Embedding Similarity
- (nếu cần) Attention Rollout

### Ưu tiên 4 (giải thích quyết định)
- Saliency Map

---


---



---

