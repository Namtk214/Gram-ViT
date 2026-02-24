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

    # Compute gradient norm before updates
    grad_norm = tree_norm(g)

    updates, opt_state = tx.update(g, opt_state)
    params = optax.apply_updates(params, updates)
    l = jax.lax.pmean(l, axis_name='batch')
    grad_norm = jax.lax.pmean(grad_norm, axis_name='batch')

    return params, opt_state, l, new_rng, grad_norm

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
               'Mixer': models.MlpMixer}[config.get('model_type', 'ViT')]
  model = model_cls(num_classes=dataset_info['num_classes'], **config.model)

  def init_model():
    return model.init(
        jax.random.PRNGKey(0),
        # Discard the "num_local_devices" dimension for initialization.
        jnp.ones(batch['image'].shape[1:], batch['image'].dtype.name),
        train=False)

  # Use JIT to make sure params reside in CPU memory.
  variables = jax.jit(init_model, backend='cpu')()

  # Check if we should load pretrained weights
  train_from_scratch = config.get('train_from_scratch', False)

  if train_from_scratch:
    # Training from scratch - use randomly initialized parameters
    logging.info('🚀 Training from scratch (no pretrained weights)')
    params = variables['params']
  else:
    # Try to load pretrained weights
    model_or_filename = config.get('model_or_filename')
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
      params_repl, opt_state_repl, loss_repl, update_rng_repl, grad_norm_repl = update_fn_repl(
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
      grad_norm = float(flax.jax_utils.unreplicate(grad_norm_repl))
      param_norm = float(tree_norm(flax.jax_utils.unreplicate(params_repl)))
      current_lr = float(lr_fn(step))

      writer.write_scalars(
          step,
          dict(
              train_loss=train_loss,
              img_sec_core_train=img_sec_core_train))

      # W&B logging - Priority 1 & 2 train metrics
      if use_wandb:
        wandb.log({
            'Train/loss': train_loss,
            'Train/learning_rate': current_lr,
            'Optim/lr': current_lr,
            'Optim/grad_global_norm': grad_norm,
            'Optim/param_global_norm': param_norm,
            'System/img_sec_core_train': img_sec_core_train,
            'step': step
        }, step=step)

      done = step / total_steps
      logging.info(f'Step: {step}/{total_steps} {100*done:.1f}%, '  # pylint: disable=logging-fstring-interpolation
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

            # Log activation and Gram-lowrank stats per block
            # Traverse the intermediates tree to find encoder blocks
            if 'Transformer' in intermediates:
              transformer_intermediates = intermediates['Transformer']
              logging.info('✓ Found Transformer in intermediates with %d keys', len(transformer_intermediates))
              logging.info('Transformer keys: %s', list(transformer_intermediates.keys()))

              encoder_block_count = 0
              for block_name, block_intermediates in transformer_intermediates.items():
                if 'encoderblock_' in block_name:
                  encoder_block_count += 1
                  block_idx = block_name.split('_')[-1]
                  logging.info('Processing block %s, available keys: %s', block_name, list(block_intermediates.keys()))

                  # Log MHSA and MLP activation stats
                  activation_found = 0
                  if 'mhsa_out_mean' in block_intermediates:
                    activation_metrics[f'Activations/block_{block_idx}/mhsa_out_mean'] = float(
                        block_intermediates['mhsa_out_mean'][0])
                    activation_found += 1
                  if 'mhsa_out_std' in block_intermediates:
                    activation_metrics[f'Activations/block_{block_idx}/mhsa_out_std'] = float(
                        block_intermediates['mhsa_out_std'][0])
                    activation_found += 1
                  if 'mlp_out_mean' in block_intermediates:
                    activation_metrics[f'Activations/block_{block_idx}/mlp_out_mean'] = float(
                        block_intermediates['mlp_out_mean'][0])
                    activation_found += 1
                  if 'mlp_out_std' in block_intermediates:
                    activation_metrics[f'Activations/block_{block_idx}/mlp_out_std'] = float(
                        block_intermediates['mlp_out_std'][0])
                    activation_found += 1

                  if activation_found > 0:
                    logging.info('  ✓ Collected %d activation metrics for block_%s', activation_found, block_idx)

                  # Log Gram-lowrank metrics if available
                  if 'GramLowRankMHSAResidual_0' in block_intermediates:
                    gram_intermediates = block_intermediates['GramLowRankMHSAResidual_0']
                    logging.info('  ✓ Found GramLowRankMHSAResidual_0, keys: %s', list(gram_intermediates.keys()))
                    gram_found = 0
                    if 'T_norm' in gram_intermediates:
                      activation_metrics[f'GramLowRank/block_{block_idx}/T_norm'] = float(
                          gram_intermediates['T_norm'][0])
                      gram_found += 1
                    if 'Z_norm' in gram_intermediates:
                      activation_metrics[f'GramLowRank/block_{block_idx}/Z_norm'] = float(
                          gram_intermediates['Z_norm'][0])
                      gram_found += 1
                    if 'T_over_Z_norm' in gram_intermediates:
                      activation_metrics[f'GramLowRank/block_{block_idx}/T_over_Z_norm'] = float(
                          gram_intermediates['T_over_Z_norm'][0])
                      gram_found += 1
                    if 'A_norm' in gram_intermediates:
                      activation_metrics[f'GramLowRank/block_{block_idx}/A_norm'] = float(
                          gram_intermediates['A_norm'][0])
                      gram_found += 1
                    if 'B_norm' in gram_intermediates:
                      activation_metrics[f'GramLowRank/block_{block_idx}/B_norm'] = float(
                          gram_intermediates['B_norm'][0])
                      gram_found += 1

                    if gram_found > 0:
                      logging.info('  ✓ Collected %d Gram-lowrank metrics for block_%s', gram_found, block_idx)
                  else:
                    logging.warning('  ✗ No GramLowRankMHSAResidual_0 found in block_%s', block_idx)

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
          else:
            logging.warning('✗ No intermediates found in state. Available keys: %s', state.keys())

        except Exception as e:
          logging.error('Failed to capture intermediates for logging: %s', str(e))
          import traceback
          logging.error('Traceback: %s', traceback.format_exc())

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
