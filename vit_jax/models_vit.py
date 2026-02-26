# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Optional, Tuple, Type

import flax.linen as nn
import jax.numpy as jnp

from vit_jax import models_resnet


Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


class IdentityLayer(nn.Module):
  """Identity layer, convenient for giving a name to an array."""

  @nn.compact
  def __call__(self, x):
    return x


class AddPositionEmbs(nn.Module):
  """Adds learned positional embeddings to the inputs.

  Attributes:
    posemb_init: positional embedding initializer.
  """

  posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]
  param_dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs):
    """Applies the AddPositionEmbs module.

    Args:
      inputs: Inputs to the layer.

    Returns:
      Output tensor with shape `(bs, timesteps, in_dim)`.
    """
    # inputs.shape is (batch_size, seq_len, emb_dim).
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
    pe = self.param(
        'pos_embedding', self.posemb_init, pos_emb_shape, self.param_dtype)
    return inputs + pe


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  mlp_dim: int
  dtype: Dtype = jnp.float32
  param_dtype: Dtype = jnp.float32
  out_dim: Optional[int] = None
  dropout_rate: float = 0.1
  kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.xavier_uniform()
  bias_init: Callable[[PRNGKey, Shape, Dtype],
                      Array] = nn.initializers.normal(stddev=1e-6)

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        features=self.mlp_dim,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(  # pytype: disable=wrong-arg-types
            inputs)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(
        features=actual_out_dim,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(  # pytype: disable=wrong-arg-types
            x)
    output = nn.Dropout(
        rate=self.dropout_rate)(
            output, deterministic=deterministic)
    return output


class HeadWiseGramLowRankBranch(nn.Module):
  """Head-wise Gram Low-Rank Branch for MHSA output with RMSNorm.

  Instead of using a single A, B pair for the entire hidden dimension D,
  this module uses separate A^(h), B^(h) pairs for each attention head.

  For each head h:
    G_t^(h) = (X^(h) @ X^(h).T) / d_h  (token Gram matrix for head h)
    T_g^(h) = G_t^(h) @ (A^(h) @ B^(h))  (correction for head h)

  Then concatenate all T_g^(h) and add to MHSA output Z.

  Attributes:
    num_heads: Number of attention heads.
    rank: Rank for low-rank matrices A and B (per head).
    param_dtype: Data type for parameters.
    eps: Epsilon for RMSNorm stability.
  """

  num_heads: int
  rank: int = 64
  param_dtype: Dtype = jnp.float32
  eps: float = 1e-6

  @nn.compact
  def __call__(self, x_mhsa_in, z_mhsa_out):
    """Apply Head-wise Gram + Low-Rank correction.

    Args:
      x_mhsa_in: Input to MHSA (after LayerNorm), shape [B, N, D].
      z_mhsa_out: Output from MHSA, shape [B, N, D].

    Returns:
      Y = Z + RMSNorm(T_g), where T_g is the concatenated correction, shape [B, N, D].
    """
    # Get shapes
    batch_size, num_tokens, hidden_dim = x_mhsa_in.shape
    d_h = hidden_dim // self.num_heads  # dimension per head

    # Step 1: Apply RMSNorm to X before computing Gram matrix
    x_rms = jnp.sqrt(jnp.mean(x_mhsa_in ** 2, axis=-1, keepdims=True) + self.eps)
    x_scale = self.param('X_scale', nn.initializers.ones, (1, 1, hidden_dim), self.param_dtype)
    x_normed = (x_mhsa_in / x_rms) * x_scale

    # Step 2: Reshape to [B, N, H, d_h] for head-wise processing
    x_heads = x_normed.reshape(batch_size, num_tokens, self.num_heads, d_h)

    # Step 3: Process each head separately
    head_corrections = []
    for h in range(self.num_heads):
      # Get head h: [B, N, d_h]
      x_h = x_heads[:, :, h, :]

      # Compute token Gram matrix for head h: [B, N, N]
      gram_h = jnp.matmul(x_h, jnp.transpose(x_h, (0, 2, 1))) / d_h

      # Low-rank parameters for head h
      # A^(h): [N, r], initialized with He/Kaiming uniform
      # B^(h): [r, d_h], initialized with zeros (makes branch no-op initially)
      a_matrix_h = self.param(
          f'A_head_{h}',
          nn.initializers.he_uniform(),
          (num_tokens, self.rank),
          self.param_dtype
      )

      b_matrix_h = self.param(
          f'B_head_{h}',
          nn.initializers.zeros,
          (self.rank, d_h),
          self.param_dtype
      )

      # Compute P^(h) = A^(h) @ B^(h), shape [N, d_h]
      p_matrix_h = jnp.matmul(a_matrix_h, b_matrix_h)

      # Compute correction term T_g^(h) = G_t^(h) @ P^(h), shape [B, N, d_h]
      t_g_h = jnp.matmul(gram_h, p_matrix_h)

      head_corrections.append(t_g_h)

    # Step 4: Concatenate all head corrections along channel dimension
    # Each t_g_h is [B, N, d_h], concat to get [B, N, D]
    t_g = jnp.concatenate(head_corrections, axis=-1)

    # Step 5: Apply RMSNorm to correction term T_g before adding to Z
    t_rms = jnp.sqrt(jnp.mean(t_g ** 2, axis=-1, keepdims=True) + self.eps)
    t_scale = self.param('T_scale', nn.initializers.ones, (1, 1, hidden_dim), self.param_dtype)
    t_normed = (t_g / t_rms) * t_scale

    # Step 6: Add normalized correction to MHSA output
    return z_mhsa_out + t_normed


class StyleRepresentationBranch(nn.Module):
  """Style Representation Branch using Channel Gram matrix with Low-Rank.

  Uses channel Gram matrix S = X^T @ X / N (shape [B, d, d]) instead of token Gram.
  Applies low-rank decomposition with matrices C and D to create style correction.

  Formula:
    S = (X^T @ X) / N                    (channel Gram, [B, d, d])
    T_style = (S @ C @ D^T)^T            (style correction, [B, N, d])
    output = RMSNorm(T_style)

  Attributes:
    rank: Rank for low-rank matrices C and D.
    param_dtype: Data type for parameters.
    eps: Epsilon for RMSNorm stability.
  """

  rank: int = 64
  param_dtype: Dtype = jnp.float32
  eps: float = 1e-6

  @nn.compact
  def __call__(self, x_for_gram):
    """Apply Style Representation Branch with channel Gram.

    Args:
      x_for_gram: Input for computing Gram (typically x_ln after LayerNorm), shape [B, N, d].

    Returns:
      T_style after RMSNorm, shape [B, N, d].
    """
    # Get shapes
    batch_size, num_tokens, hidden_dim = x_for_gram.shape

    # Step 1: Compute channel Gram matrix S = X^T @ X / N
    # X: [B, N, d], X^T: [B, d, N]
    # S: [B, d, d]
    channel_gram = jnp.matmul(
        jnp.transpose(x_for_gram, (0, 2, 1)),  # [B, d, N]
        x_for_gram                              # [B, N, d]
    ) / num_tokens                              # [B, d, d]

    # Step 2: Low-rank parameters for style branch
    # C: [d, r_s], initialized with small normal (LoRA-style)
    # D: [N, r_s], initialized with zeros (makes branch no-op initially)
    c_matrix = self.param(
        'C',
        nn.initializers.normal(stddev=1e-2),
        (hidden_dim, self.rank),
        self.param_dtype
    )

    d_matrix = self.param(
        'D',
        nn.initializers.zeros,
        (num_tokens, self.rank),
        self.param_dtype
    )

    # Step 3: Compute style correction
    # S @ C: [B, d, d] @ [d, r_s] -> [B, d, r_s]
    sc = jnp.matmul(channel_gram, c_matrix)

    # (S @ C) @ D^T: [B, d, r_s] @ [r_s, N] -> [B, d, N]
    scd_t = jnp.matmul(sc, jnp.transpose(d_matrix, (1, 0)))

    # Transpose to get T_style: [B, N, d]
    t_style = jnp.transpose(scd_t, (0, 2, 1))

    # Step 4: Apply RMSNorm to style correction
    t_rms = jnp.sqrt(jnp.mean(t_style ** 2, axis=-1, keepdims=True) + self.eps)
    t_scale = self.param('T_scale', nn.initializers.ones, (1, 1, hidden_dim), self.param_dtype)
    t_style_normed = (t_style / t_rms) * t_scale

    return t_style_normed


class GramLowRankMHSAResidual(nn.Module):
  """Gram + Low-Rank residual correction for MHSA output with RMSNorm.

  Adds a correction term to MHSA output:
    Y = Z + RMSNorm(T), where T = G_t(AB)
    G_t = RMSNorm(X) @ RMSNorm(X)^T / D (token Gram matrix with normalized input)

  Attributes:
    rank: Rank for low-rank matrices A and B.
    param_dtype: Data type for parameters.
    eps: Epsilon for RMSNorm stability.
  """

  rank: int = 64
  param_dtype: Dtype = jnp.float32
  eps: float = 1e-6

  @nn.compact
  def __call__(self, x_mhsa_in, z_mhsa_out):
    """Apply Gram + Low-Rank residual correction.

    Args:
      x_mhsa_in: Input to MHSA (after LayerNorm), shape [B, N, D].
      z_mhsa_out: Output from MHSA, shape [B, N, D].

    Returns:
      Y = Z + RMSNorm(T), where T is the correction term, shape [B, N, D].
    """
    # Get shapes
    batch_size, num_tokens, hidden_dim = x_mhsa_in.shape

    # Step 1: Apply RMSNorm to X before computing Gram matrix
    # RMSNorm: normalize across hidden_dim dimension (per token)
    x_rms = jnp.sqrt(jnp.mean(x_mhsa_in ** 2, axis=-1, keepdims=True) + self.eps)
    x_scale = self.param('X_scale', nn.initializers.ones, (1, 1, hidden_dim), self.param_dtype)
    x_normed = (x_mhsa_in / x_rms) * x_scale

    # Step 2: Compute token Gram matrix G_t = X_normed @ X_normed^T / D
    # Shape: [B, N, N]
    gram_matrix = jnp.matmul(x_normed, jnp.transpose(x_normed, (0, 2, 1))) / hidden_dim

    # Step 3: Define low-rank parameters A and B
    # A: [N, r], initialized with He/Kaiming uniform
    # B: [r, D], initialized with zeros (makes branch no-op initially)
    a_matrix = self.param(
        'A',
        nn.initializers.he_uniform(),
        (num_tokens, self.rank),
        self.param_dtype
    )

    b_matrix = self.param(
        'B',
        nn.initializers.zeros,
        (self.rank, hidden_dim),
        self.param_dtype
    )

    # Step 4: Compute P = AB, shape [N, D]
    p_matrix = jnp.matmul(a_matrix, b_matrix)

    # Step 5: Compute correction term T = G_t @ P, shape [B, N, D]
    correction_term = jnp.matmul(gram_matrix, p_matrix)

    # Step 6: Apply RMSNorm to correction term T before adding to Z
    # RMSNorm: normalize across hidden_dim dimension (per token)
    t_rms = jnp.sqrt(jnp.mean(correction_term ** 2, axis=-1, keepdims=True) + self.eps)
    t_scale = self.param('T_scale', nn.initializers.ones, (1, 1, hidden_dim), self.param_dtype)
    t_normed = (correction_term / t_rms) * t_scale

    # Compute metrics for W&B logging
    t_norm = jnp.linalg.norm(correction_term)
    t_normed_norm = jnp.linalg.norm(t_normed)
    z_norm = jnp.linalg.norm(z_mhsa_out)
    t_over_z_norm = t_norm / (z_norm + 1e-8)
    x_norm = jnp.linalg.norm(x_mhsa_in)
    x_normed_norm = jnp.linalg.norm(x_normed)
    a_norm = jnp.linalg.norm(a_matrix)
    b_norm = jnp.linalg.norm(b_matrix)

    # Sow metrics for logging (stored in 'intermediates' collection)
    self.sow('intermediates', 'T_norm', t_norm)
    self.sow('intermediates', 'T_normed_norm', t_normed_norm)
    self.sow('intermediates', 'Z_norm', z_norm)
    self.sow('intermediates', 'T_over_Z_norm', t_over_z_norm)
    self.sow('intermediates', 'X_norm', x_norm)
    self.sow('intermediates', 'X_normed_norm', x_normed_norm)
    self.sow('intermediates', 'A_norm', a_norm)
    self.sow('intermediates', 'B_norm', b_norm)

    # Step 7: Add normalized correction to MHSA output
    return z_mhsa_out + t_normed


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    inputs: input data.
    mlp_dim: dimension of the mlp on top of attention block.
    dtype: the dtype of the computation (default: float32).
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout for attention heads.
    deterministic: bool, deterministic or not (to apply dropout).
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
    use_gram_lowrank_mhsa: Enable Gram + Low-Rank residual for MHSA (original, token Gram).
    gram_lowrank_rank: Rank for low-rank matrices in Gram residual (original).
    use_headwise_gram_lowrank: Enable Head-wise Gram Low-Rank branch (token Gram per head).
    headwise_gram_rank: Rank for low-rank matrices in Head-wise Gram branch.
    use_style_branch: Enable Style Representation Branch (channel Gram).
    style_rank: Rank for low-rank matrices in Style branch.
  """

  mlp_dim: int
  num_heads: int
  dtype: Dtype = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  # Original Gram-lowrank (whole D)
  use_gram_lowrank_mhsa: bool = False
  gram_lowrank_rank: int = 64
  # New: Head-wise Gram-lowrank
  use_headwise_gram_lowrank: bool = False
  headwise_gram_rank: int = 64
  # New: Style Representation Branch (Channel Gram)
  use_style_branch: bool = False
  style_rank: int = 64

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """Applies Encoder1DBlock module.

    Args:
      inputs: Inputs to the layer.
      deterministic: Dropout will not be applied when set to true.

    Returns:
      output after transformer encoder block.
    """

    # Attention block.
    assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'

    # Cannot use both Gram-lowrank variants simultaneously
    assert not (self.use_gram_lowrank_mhsa and self.use_headwise_gram_lowrank), \
        "Cannot enable both use_gram_lowrank_mhsa and use_headwise_gram_lowrank at the same time"

    # LayerNorm before MHSA
    x_mhsa_in = nn.LayerNorm(dtype=self.dtype)(inputs)

    # MHSA output
    z_mhsa_out = nn.MultiHeadDotProductAttention(
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads)(
            x_mhsa_in, x_mhsa_in)

    # Apply Gram + Low-Rank residual (original version - whole D)
    # This gives us u = z + t_gram
    u = z_mhsa_out
    if self.use_gram_lowrank_mhsa:
      u = GramLowRankMHSAResidual(
          rank=self.gram_lowrank_rank,
          param_dtype=self.dtype)(
              x_mhsa_in, z_mhsa_out)

    # Apply Head-wise Gram + Low-Rank branch (new version - per head)
    if self.use_headwise_gram_lowrank:
      u = HeadWiseGramLowRankBranch(
          num_heads=self.num_heads,
          rank=self.headwise_gram_rank,
          param_dtype=self.dtype)(
              x_mhsa_in, z_mhsa_out)

    # Apply LayerNorm on u (attention output after token Gram if enabled)
    # This is the new LN required for style branch integration
    u_bar = u
    if self.use_style_branch:
      u_bar = nn.LayerNorm(dtype=self.dtype)(u)

      # Compute style correction using channel Gram
      t_style = StyleRepresentationBranch(
          rank=self.style_rank,
          param_dtype=self.dtype)(
              x_mhsa_in)

      # Add style correction to normalized attention output
      y = u_bar + t_style
    else:
      # No style branch, use u directly
      y = u_bar

    x = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate)(
            y, deterministic=deterministic)

    # Compute block output
    block_output = x + y

    # Sow block output statistics (mean, abs_mean, std across all tokens and dimensions)
    self.sow('intermediates', 'block_output_mean', jnp.mean(block_output))
    self.sow('intermediates', 'block_output_abs_mean', jnp.mean(jnp.abs(block_output)))
    self.sow('intermediates', 'block_output_std', jnp.std(block_output))

    return block_output


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation.

  Attributes:
    num_layers: number of layers
    mlp_dim: dimension of the mlp on top of attention block
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout rate in self attention.
    use_gram_lowrank_mhsa: Enable Gram + Low-Rank residual for MHSA (original, token Gram).
    gram_lowrank_rank: Rank for low-rank matrices in Gram residual (original).
    use_headwise_gram_lowrank: Enable Head-wise Gram Low-Rank branch (token Gram per head).
    headwise_gram_rank: Rank for low-rank matrices in Head-wise Gram branch.
    use_style_branch: Enable Style Representation Branch (channel Gram).
    style_rank: Rank for low-rank matrices in Style branch.
  """

  num_layers: int
  mlp_dim: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  add_position_embedding: bool = True
  # Original Gram-lowrank (token Gram)
  use_gram_lowrank_mhsa: bool = False
  gram_lowrank_rank: int = 64
  # New: Head-wise Gram-lowrank (token Gram per head)
  use_headwise_gram_lowrank: bool = False
  headwise_gram_rank: int = 64
  # New: Style Representation Branch (channel Gram)
  use_style_branch: bool = False
  style_rank: int = 64

  @nn.compact
  def __call__(self, x, *, train):
    """Applies Transformer model on the inputs.

    Args:
      x: Inputs to the layer.
      train: Set to `True` when training.

    Returns:
      output of a transformer encoder.
    """
    assert x.ndim == 3  # (batch, len, emb)

    if self.add_position_embedding:
      x = AddPositionEmbs(
          posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
          name='posembed_input')(
              x)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    # Input Encoder
    for lyr in range(self.num_layers):
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          use_gram_lowrank_mhsa=self.use_gram_lowrank_mhsa,
          gram_lowrank_rank=self.gram_lowrank_rank,
          use_headwise_gram_lowrank=self.use_headwise_gram_lowrank,
          headwise_gram_rank=self.headwise_gram_rank,
          use_style_branch=self.use_style_branch,
          style_rank=self.style_rank,
          name=f'encoderblock_{lyr}',
          num_heads=self.num_heads)(
              x, deterministic=not train)
    encoded = nn.LayerNorm(name='encoder_norm')(x)

    return encoded


class VisionTransformer(nn.Module):
  """VisionTransformer."""

  num_classes: int
  patches: Any
  transformer: Any
  hidden_size: int
  resnet: Optional[Any] = None
  representation_size: Optional[int] = None
  classifier: str = 'token'
  head_bias_init: float = 0.
  encoder: Type[nn.Module] = Encoder
  model_name: Optional[str] = None

  @nn.compact
  def __call__(self, inputs, *, train):

    x = inputs
    # (Possibly partial) ResNet root.
    if self.resnet is not None:
      width = int(64 * self.resnet.width_factor)

      # Root block.
      x = models_resnet.StdConv(
          features=width,
          kernel_size=(7, 7),
          strides=(2, 2),
          use_bias=False,
          name='conv_root')(
              x)
      x = nn.GroupNorm(name='gn_root')(x)
      x = nn.relu(x)
      x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')

      # ResNet stages.
      if self.resnet.num_layers:
        x = models_resnet.ResNetStage(
            block_size=self.resnet.num_layers[0],
            nout=width,
            first_stride=(1, 1),
            name='block1')(
                x)
        for i, block_size in enumerate(self.resnet.num_layers[1:], 1):
          x = models_resnet.ResNetStage(
              block_size=block_size,
              nout=width * 2**i,
              first_stride=(2, 2),
              name=f'block{i + 1}')(
                  x)

    n, h, w, c = x.shape

    # We can merge s2d+emb into a single conv; it's the same.
    x = nn.Conv(
        features=self.hidden_size,
        kernel_size=self.patches.size,
        strides=self.patches.size,
        padding='VALID',
        name='embedding')(
            x)

    # Here, x is a grid of embeddings.

    # (Possibly partial) Transformer.
    if self.transformer is not None:
      n, h, w, c = x.shape
      x = jnp.reshape(x, [n, h * w, c])

      # If we want to add a class token, add it here.
      if self.classifier in ['token', 'token_unpooled']:
        cls = self.param('cls', nn.initializers.zeros, (1, 1, c))
        cls = jnp.tile(cls, [n, 1, 1])
        x = jnp.concatenate([cls, x], axis=1)

      x = self.encoder(name='Transformer', **self.transformer)(x, train=train)

    if self.classifier == 'token':
      x = x[:, 0]
    elif self.classifier == 'gap':
      x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))  # (1,) or (1,2)
    elif self.classifier in ['unpooled', 'token_unpooled']:
      pass
    else:
      raise ValueError(f'Invalid classifier={self.classifier}')

    if self.representation_size is not None:
      x = nn.Dense(features=self.representation_size, name='pre_logits')(x)
      x = nn.tanh(x)
    else:
      x = IdentityLayer(name='pre_logits')(x)

    if self.num_classes:
      x = nn.Dense(
          features=self.num_classes,
          name='head',
          kernel_init=nn.initializers.zeros,
          bias_init=nn.initializers.constant(self.head_bias_init))(x)
    return x
