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

import functools
import os
import time

from absl import logging
from clu import metric_writers
from clu import periodic_actions
import flax
from flax.training import checkpoints as flax_checkpoints
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow as tf
import wandb
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from vit_jax import checkpoint
from vit_jax import input_pipeline
from vit_jax import models
from vit_jax import utils


def compute_confusion_matrix(y_true, y_pred, num_classes=10):
  """Compute confusion matrix from true and predicted labels."""
  conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
  for t, p in zip(y_true, y_pred):
    conf_matrix[t, p] += 1
  return conf_matrix


def compute_per_class_accuracy(y_true, y_pred, class_names):
  """Compute per-class accuracy.

  Args:
    y_true: True labels (class indices).
    y_pred: Predicted labels (class indices).
    class_names: List of class names.

  Returns:
    Dictionary mapping class names to their accuracies.
  """
  per_class_acc = {}
  num_classes = len(class_names)
  for i in range(num_classes):
    mask = y_true == i
    if mask.sum() > 0:
      per_class_acc[class_names[i]] = (y_pred[mask] == i).sum() / mask.sum()
    else:
      per_class_acc[class_names[i]] = 0.0
  return per_class_acc


def compute_topk_accuracy(logits, labels, k=5):
  """Compute top-k accuracy."""
  top_k_preds = np.argsort(logits, axis=-1)[:, -k:]
  true_labels = np.argmax(labels, axis=-1)
  correct = np.array([label in preds for label, preds in zip(true_labels, top_k_preds)])
  return correct.mean()


def tree_norm(tree):
  """Compute global norm of a pytree."""
  return jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree.leaves(tree)))


def compute_saliency_map(model_apply, params, image, true_class_idx):
  """Compute saliency map: gradient of predicted class logit wrt input image.

  Args:
    model_apply: Model apply function
    params: Model parameters
    image: Input image [H, W, C]
    true_class_idx: True class index for computing gradient

  Returns:
    Saliency map [H, W] - absolute value of gradient magnitude
  """
  def loss_fn(img):
    logits = model_apply({'params': params}, img[None, ...], train=False)
    return logits[0, true_class_idx]

  grad_fn = jax.grad(loss_fn)
  gradient = grad_fn(image)

  # Take absolute value and sum across channels to get [H, W] saliency map
  saliency = jnp.abs(gradient).sum(axis=-1)

  # Normalize to [0, 1]
  saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

  return saliency


def create_saliency_visualization(image, saliency_map, pred_class_name, true_class_name):
  """Create visualization of image and saliency map side by side.

  Args:
    image: Original image [H, W, C], values in [0, 1]
    saliency_map: Saliency map [H, W], values in [0, 1]
    pred_class_name: Predicted class name
    true_class_name: True class name

  Returns:
    Matplotlib figure
  """
  fig, axes = plt.subplots(1, 2, figsize=(8, 4))

  # Original image
  axes[0].imshow(np.array(image))
  axes[0].set_title(f'True: {true_class_name}\nPred: {pred_class_name}')
  axes[0].axis('off')

  # Saliency map
  im = axes[1].imshow(np.array(saliency_map), cmap='hot')
  axes[1].set_title('Saliency Map')
  axes[1].axis('off')
  plt.colorbar(im, ax=axes[1])

  plt.tight_layout()
  return fig


# DISABLED to save memory - Histogram logging is expensive
# def log_histograms(params, grads, prefix, step, sample_rate=0.1):
#   """Log histograms of parameters and gradients to W&B.
#
#   Args:
#     params: Parameter tree
#     grads: Gradient tree
#     prefix: Prefix for metric names
#     step: Current training step
#     sample_rate: Fraction of params to log (to avoid overhead)
#
#   Returns:
#     Dictionary of histogram metrics for W&B
#   """
#   metrics = {}
#   param_flat = jax.tree_util.tree_leaves(params)
#   grad_flat = jax.tree_util.tree_leaves(grads)
#   param_names = [f'layer_{i}' for i in range(len(param_flat))]
#   num_to_log = max(1, int(len(param_flat) * sample_rate))
#   indices = np.linspace(0, len(param_flat) - 1, num_to_log, dtype=int)
#   for idx in indices:
#     name = param_names[idx]
#     param_array = np.array(param_flat[idx]).flatten()
#     metrics[f'{prefix}/Params/{name}'] = wandb.Histogram(param_array)
#     grad_array = np.array(grad_flat[idx]).flatten()
#     metrics[f'{prefix}/Grads/{name}'] = wandb.Histogram(grad_array)
#   return metrics


def make_update_fn(*, apply_fn, accum_steps, tx):
  """Returns update step for data parallel training."""

  def update_fn(params, opt_state, batch, rng):

    _, new_rng = jax.random.split(rng)
    # Bind the rng key to the device id (which is unique across hosts)
    # Note: This is only used for multi-host training (i.e. multiple computers
    # each with multiple accelerators).
    dropout_rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))

    def cross_entropy_loss(*, logits, labels):
      logp = jax.nn.log_softmax(logits)
      return -jnp.mean(jnp.sum(logp * labels, axis=1))

    def loss_fn(params, images, labels):
      logits = apply_fn(
          dict(params=params),
          rngs=dict(dropout=dropout_rng),
          inputs=images,
          train=True)
      return cross_entropy_loss(logits=logits, labels=labels)

    l, g = utils.accumulate_gradient(
        jax.value_and_grad(loss_fn), params, batch['image'], batch['label'],
        accum_steps)
    g = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name='batch'), g)

    # Compute accuracy with a separate forward pass (without dropout for consistent accuracy)
    logits = apply_fn(
        dict(params=params),
        inputs=batch['image'],
        train=False)
    preds = jnp.argmax(logits, axis=-1)
    labels_idx = jnp.argmax(batch['label'], axis=-1)
    correct = jnp.equal(preds, labels_idx)
    accuracy = jnp.mean(correct)

    # Compute gradient norm before updates
    grad_norm = tree_norm(g)

    updates, opt_state = tx.update(g, opt_state)
    params = optax.apply_updates(params, updates)
    l = jax.lax.pmean(l, axis_name='batch')
    accuracy = jax.lax.pmean(accuracy, axis_name='batch')
    grad_norm = jax.lax.pmean(grad_norm, axis_name='batch')

    return params, opt_state, l, new_rng, grad_norm, accuracy

  return jax.pmap(update_fn, axis_name='batch', donate_argnums=(0,))


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
  """Runs training interleaved with evaluation."""

  # Initialize Weights & Biases
  # Will use existing API key from environment (WANDB_API_KEY) or ~/.netrc
  try:
    wandb.init(
        project="gram-vit-cifar10",
        name=f"{config.model.model_name}_{config.dataset}",
        config=config.to_dict(),
        dir=workdir,
        resume="allow"  # Allow resuming if run with same name
    )
    use_wandb = True
    logging.info("W&B initialized successfully")
  except Exception as e:
    logging.warning(f"W&B initialization failed: {e}. Running without W&B logging.")
    use_wandb = False

  # Setup input pipeline
  dataset_info = input_pipeline.get_dataset_info(config.dataset, 'train')

  ds_train, ds_test = input_pipeline.get_datasets(config)
  batch = next(iter(ds_train))
  logging.info(ds_train)
  logging.info(ds_test)

  # Build VisionTransformer architecture
  model_cls = {'ViT': models.VisionTransformer,
               'Mixer': models.MlpMixer}[getattr(config, 'model_type', 'ViT')]
  model = model_cls(num_classes=dataset_info['num_classes'], **config.model)

  def init_model():
    return model.init(
        jax.random.PRNGKey(0),
        # Discard the "num_local_devices" dimension for initialization.
        jnp.ones(batch['image'].shape[1:], batch['image'].dtype.name),
        train=False)

  # Use JIT to make sure params reside in CPU memory.
  variables = jax.jit(init_model, backend='cpu')()

  # Log Gram-lowrank configuration
  use_gram_lowrank = getattr(config.model.transformer, 'use_gram_lowrank_mhsa', False)

  # Handle string "True"/"False" from command line
  if isinstance(use_gram_lowrank, str):
    use_gram_lowrank = use_gram_lowrank.lower() in ('true', '1', 'yes')
    logging.warning('Config use_gram_lowrank_mhsa received as string, converted to: %s', use_gram_lowrank)

  if use_gram_lowrank:
    gram_rank = getattr(config.model.transformer, 'gram_lowrank_rank', 8)
    logging.info('✓ Gram-LowRank ENABLED: rank=%d', gram_rank)
  else:
    logging.info('✗ Gram-LowRank DISABLED (value was: %s, type: %s)',
                 getattr(config.model.transformer, 'use_gram_lowrank_mhsa', False),
                 type(getattr(config.model.transformer, 'use_gram_lowrank_mhsa', False)))

  # Check if we should load pretrained weights
  train_from_scratch = getattr(config, 'train_from_scratch', False)

  if train_from_scratch:
    # Training from scratch - use randomly initialized parameters
    logging.info('🚀 Training from scratch (no pretrained weights)')
    params = variables['params']
  else:
    # Try to load pretrained weights
    model_or_filename = getattr(config, 'model_or_filename', None)
    if model_or_filename:
      # Loading model from repo published with  "How to train your ViT? Data,
      # Augmentation, and Regularization in Vision Transformers" paper.
      # https://arxiv.org/abs/2106.10270
      if '-' in model_or_filename:
        filename = model_or_filename
      else:
        # Select best checkpoint from i21k pretraining by final upstream
        # validation accuracy.
        df = checkpoint.get_augreg_df(directory=config.pretrained_dir)
        sel = df.filename.apply(
            lambda filename: filename.split('-')[0] == model_or_filename)
        best = df.loc[sel].query('ds=="i21k"').sort_values('final_val').iloc[-1]
        filename = best.filename
        logging.info('Selected fillename="%s" for "%s" with final_val=%.3f',
                     filename, model_or_filename, best.final_val)
      pretrained_path = os.path.join(config.pretrained_dir,
                                     f'{config.model.model_name}.npz')
    else:
      # ViT / Mixer papers
      filename = config.model.model_name

    pretrained_path = os.path.join(config.pretrained_dir, f'{filename}.npz')

    # Check if pretrained file exists
    if tf.io.gfile.exists(pretrained_path):
      logging.info('Loading pretrained weights from "%s"', pretrained_path)
      params = checkpoint.load_pretrained(
          pretrained_path=pretrained_path,
          init_params=variables['params'],
          model_config=config.model)
    else:
      # File doesn't exist - warn and train from scratch
      logging.warning(
          'Pretrained weights not found at "%s". Training from scratch instead.',
          pretrained_path)
      logging.warning(
          'To download pretrained weights: '
          'gsutil cp gs://vit_models/imagenet21k/%s.npz %s',
          config.model.model_name, pretrained_path)
      params = variables['params']

  total_steps = config.total_steps

  lr_fn = utils.create_learning_rate_schedule(total_steps, config.base_lr,
                                              config.decay_type,
                                              config.warmup_steps)
  tx = optax.chain(
      optax.clip_by_global_norm(config.grad_norm_clip),
      optax.sgd(
          learning_rate=lr_fn,
          momentum=0.9,
          accumulator_dtype='bfloat16',
      ),
  )

  update_fn_repl = make_update_fn(
      apply_fn=model.apply, accum_steps=config.accum_steps, tx=tx)
  infer_fn_repl = jax.pmap(functools.partial(model.apply, train=False))

  initial_step = 1
  opt_state = tx.init(params)
  params, opt_state, initial_step = flax_checkpoints.restore_checkpoint(
      workdir, (params, opt_state, initial_step))
  logging.info('Will start/continue training at initial_step=%d', initial_step)

  params_repl, opt_state_repl = flax.jax_utils.replicate((params, opt_state))

  # Delete references to the objects that are not needed anymore
  del opt_state
  del params

  # Prepare the learning-rate and pre-fetch it to device to avoid delays.
  update_rng_repl = flax.jax_utils.replicate(jax.random.PRNGKey(0))

  # Setup metric writer & hooks.
  writer = metric_writers.create_default_writer(workdir, asynchronous=False)
  writer.write_hparams(config.to_dict())
  hooks = [
      periodic_actions.Profile(logdir=workdir),
      periodic_actions.ReportProgress(
          num_train_steps=total_steps, writer=writer),
  ]

  # Run training loop
  logging.info('Starting training loop; initial compile can take a while...')
  t0 = lt0 = time.time()
  lstep = initial_step
  for step, batch in zip(
      range(initial_step, total_steps + 1),
      input_pipeline.prefetch(ds_train, config.prefetch)):

    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      params_repl, opt_state_repl, loss_repl, update_rng_repl, grad_norm_repl, train_accuracy_repl = update_fn_repl(
          params_repl, opt_state_repl, batch, update_rng_repl)

    for hook in hooks:
      hook(step)

    if step == initial_step:
      logging.info('First step took %.1f seconds.', time.time() - t0)
      t0 = time.time()
      lt0, lstep = time.time(), step

    # Report training metrics
    if config.progress_every and step % config.progress_every == 0:
      img_sec_core_train = (config.batch * (step - lstep) /
                            (time.time() - lt0)) / jax.device_count()
      lt0, lstep = time.time(), step

      train_loss = float(flax.jax_utils.unreplicate(loss_repl))
      train_accuracy = float(flax.jax_utils.unreplicate(train_accuracy_repl))
      current_lr = float(lr_fn(step))

      writer.write_scalars(
          step,
          dict(
              train_loss=train_loss,
              train_accuracy=train_accuracy,
              img_sec_core_train=img_sec_core_train))

      # W&B logging - train metrics
      if use_wandb:
        wandb.log({
            'Train/loss': train_loss,
            'Train/accuracy': train_accuracy,
            'Train/learning_rate': current_lr,
            'Optim/lr': current_lr,
            'System/img_sec_core_train': img_sec_core_train,
            'step': step
        }, step=step)

      done = step / total_steps
      logging.info(f'Step: {step}/{total_steps} {100*done:.1f}%, '  # pylint: disable=logging-fstring-interpolation
                   f'Train loss: {train_loss:.4f}, '
                   f'Train acc: {train_accuracy:.4f}, '
                   f'img/sec/core: {img_sec_core_train:.1f}, '
                   f'ETA: {(time.time()-t0)/done*(1-done)/3600:.2f}h')

    # Run evaluation
    if ((config.eval_every and step % config.eval_every == 0) or
        (step == total_steps)):

      accuracies = []
      all_logits = []
      all_labels = []
      val_losses = []

      def cross_entropy_loss(*, logits, labels):
        logp = jax.nn.log_softmax(logits)
        return -jnp.mean(jnp.sum(logp * labels, axis=1))

      lt0 = time.time()
      for test_batch in input_pipeline.prefetch(ds_test, config.prefetch):
        logits = infer_fn_repl(
            dict(params=params_repl), test_batch['image'])

        # Flatten replicated outputs
        logits_flat = logits.reshape(-1, logits.shape[-1])
        labels_flat = test_batch['label'].reshape(-1, test_batch['label'].shape[-1])

        # Compute val loss
        val_loss = float(cross_entropy_loss(logits=logits_flat, labels=labels_flat))
        val_losses.append(val_loss)

        # Collect for metrics
        all_logits.append(logits_flat)
        all_labels.append(labels_flat)

        accuracies.append(
            (np.argmax(logits_flat, axis=-1) == np.argmax(labels_flat, axis=-1)).mean())

      # Concatenate all batches
      all_logits = np.concatenate(all_logits, axis=0)
      all_labels = np.concatenate(all_labels, axis=0)
      y_true = np.argmax(all_labels, axis=-1)
      y_pred = np.argmax(all_logits, axis=-1)

      accuracy_test = np.mean(accuracies)
      val_loss = np.mean(val_losses)
      top5_accuracy = compute_topk_accuracy(all_logits, all_labels, k=5)

      # Get class names from dataset info
      class_names = [dataset_info['int2str'](i) for i in range(dataset_info['num_classes'])]

      # Compute per-class accuracy
      per_class_acc = compute_per_class_accuracy(y_true, y_pred, class_names)

      img_sec_core_test = (
          config.batch_eval * ds_test.cardinality().numpy() /
          (time.time() - lt0) / jax.device_count())
      lt0 = time.time()

      lr = float(lr_fn(step))
      logging.info(f'Step: {step} '  # pylint: disable=logging-fstring-interpolation
                   f'Learning rate: {lr:.7f}, '
                   f'Test accuracy: {accuracy_test:0.5f}, '
                   f'Val loss: {val_loss:0.5f}, '
                   f'img/sec/core: {img_sec_core_test:.1f}')
      writer.write_scalars(
          step,
          dict(
              accuracy_test=accuracy_test,
              lr=lr,
              img_sec_core_test=img_sec_core_test))

      # W&B logging - Priority 1 validation metrics
      if use_wandb:
        wandb_metrics = {
            'Val/loss': val_loss,
            'Val/accuracy': accuracy_test,
            'Val/top1_accuracy': accuracy_test,
            'Val/top5_accuracy': top5_accuracy,
            'System/img_sec_core_test': img_sec_core_test,
            'step': step
        }

        # Add per-class accuracy
        for class_name, acc in per_class_acc.items():
          wandb_metrics[f'Val/per_class_accuracy/{class_name}'] = acc

        # Log confusion matrix
        wandb_metrics['Charts/confusion_matrix'] = wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true,
            preds=y_pred,
            class_names=class_names
        )

        wandb.log(wandb_metrics, step=step)

      # Log activation stats and Gram-lowrank metrics (Priority 2 & Gram-specific)
      # Use a single batch to capture intermediates
      if use_wandb:
        try:
          logging.info('Attempting to capture intermediates for logging...')
          sample_batch = next(iter(input_pipeline.prefetch(ds_test, 1)))
          # Run forward pass with mutable intermediates collection
          _, state = model.apply(
              {'params': flax.jax_utils.unreplicate(params_repl)},
              sample_batch['image'][0][:1],  # Single device, single sample
              train=False,
              mutable=['intermediates']
          )

          # Extract intermediates if available
          if 'intermediates' in state:
            logging.info('✓ Intermediates found in state')
            intermediates = state['intermediates']
            logging.info('Intermediates keys: %s', list(intermediates.keys()))
            activation_metrics = {}

            # Log block output statistics per block
            # Traverse the intermediates tree to find encoder blocks
            if 'Transformer' in intermediates:
              transformer_intermediates = intermediates['Transformer']
              logging.info('✓ Found Transformer in intermediates with %d keys', len(transformer_intermediates))

              encoder_block_count = 0
              for block_name, block_intermediates in transformer_intermediates.items():
                if 'encoderblock_' in block_name:
                  encoder_block_count += 1
                  block_idx = block_name.split('_')[-1]

                  # Log block output statistics (mean, abs_mean, std across all tokens and dims)
                  stats_found = 0
                  if 'block_output_mean' in block_intermediates:
                    activation_metrics[f'Activations/block_{block_idx}/output_mean'] = float(
                        block_intermediates['block_output_mean'][0])
                    stats_found += 1
                  if 'block_output_abs_mean' in block_intermediates:
                    activation_metrics[f'Activations/block_{block_idx}/output_abs_mean'] = float(
                        block_intermediates['block_output_abs_mean'][0])
                    stats_found += 1
                  if 'block_output_std' in block_intermediates:
                    activation_metrics[f'Activations/block_{block_idx}/output_std'] = float(
                        block_intermediates['block_output_std'][0])
                    stats_found += 1

                  if stats_found > 0:
                    logging.info('  ✓ Collected %d block output stats for block_%s', stats_found, block_idx)
                  else:
                    logging.warning('  ✗ No block output stats found for block_%s', block_idx)

              logging.info('✓ Processed %d encoder blocks', encoder_block_count)
            else:
              logging.warning('✗ No Transformer found in intermediates')

            if activation_metrics:
              logging.info('✓✓✓ SUCCESS: Logging %d activation/gram metrics to W&B', len(activation_metrics))
              logging.info('Metrics being logged: %s', list(activation_metrics.keys()))
              wandb.log(activation_metrics, step=step)
              logging.info('✓✓✓ Metrics successfully sent to W&B')
            else:
              logging.warning('✗✗✗ FAILED: No activation metrics collected')

            # Log Saliency Maps (every 1000 steps for fixed samples to save memory)
            if step % 1000 == 0:
              try:
                logging.info('Computing saliency maps...')
                # Get a few fixed validation samples
                num_samples = 4  # Reduced from 4 to save memory
                sample_images = sample_batch['image'][0][:num_samples]  # [num_samples, H, W, C]
                sample_labels = sample_batch['label'][0][:num_samples]  # [num_samples, num_classes]

                saliency_images = []
                params_unrepl = flax.jax_utils.unreplicate(params_repl)

                for i in range(num_samples):
                  img = sample_images[i]
                  true_label_idx = np.argmax(sample_labels[i])

                  # Get prediction
                  logits = model.apply({'params': params_unrepl}, img[None, ...], train=False)
                  pred_label_idx = np.argmax(logits[0])

                  # Compute saliency map
                  saliency = compute_saliency_map(
                      model.apply, params_unrepl, img, true_label_idx)

                  # Get class names
                  true_class_name = dataset_info['int2str'](true_label_idx)
                  pred_class_name = dataset_info['int2str'](pred_label_idx)

                  # Create visualization
                  fig = create_saliency_visualization(
                      img, saliency, pred_class_name, true_class_name)

                  # Convert to wandb Image
                  saliency_images.append(wandb.Image(fig))
                  plt.close(fig)

                # Log to W&B
                wandb.log({'Vis/saliency_maps': saliency_images}, step=step)
                logging.info('✓ Logged %d saliency maps', len(saliency_images))

              except Exception as e:
                logging.warning('Failed to log saliency maps: %s', str(e))
                import traceback
                logging.warning('Traceback: %s', traceback.format_exc())

          else:
            logging.warning('✗ No intermediates found in state. Available keys: %s', state.keys())

        except Exception as e:
          logging.error('Failed to capture intermediates for logging: %s', str(e))
          import traceback
          logging.error('Traceback: %s', traceback.format_exc())

      # Log Parameter and Gradient Histograms - DISABLED to save memory
      # if use_wandb and step % 1000 == 0 and step > 0:
      #   try:
      #     logging.info('Computing parameter and gradient histograms...')
      #     params_unrepl = flax.jax_utils.unreplicate(params_repl)
      #     grad_batch = next(iter(input_pipeline.prefetch(ds_test, 1)))
      #     def cross_entropy_loss(*, logits, labels):
      #       logp = jax.nn.log_softmax(logits)
      #       return -jnp.mean(jnp.sum(logp * labels, axis=1))
      #     def loss_fn(params):
      #       logits = model.apply({'params': params}, grad_batch['image'][0][:8], train=False)
      #       return cross_entropy_loss(logits=logits, labels=grad_batch['label'][0][:8])
      #     grads = jax.grad(loss_fn)(params_unrepl)
      #     hist_metrics = log_histograms(params_unrepl, grads, 'Histograms', step, sample_rate=0.1)
      #     wandb.log(hist_metrics, step=step)
      #     logging.info('✓ Logged %d parameter/gradient histograms', len(hist_metrics))
      #   except Exception as e:
      #     logging.warning('Failed to log param/grad histograms: %s', str(e))

    # Store checkpoint.
    if ((config.checkpoint_every and step % config.eval_every == 0) or
        step == total_steps):
      checkpoint_path = flax_checkpoints.save_checkpoint(
          workdir, (flax.jax_utils.unreplicate(params_repl),
                    flax.jax_utils.unreplicate(opt_state_repl), step), step)
      logging.info('Stored checkpoint at step %d to "%s"', step,
                   checkpoint_path)

  # Finish W&B run
  if use_wandb:
    wandb.finish()

  return flax.jax_utils.unreplicate(params_repl)
