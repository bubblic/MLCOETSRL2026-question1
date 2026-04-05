"""Model trainer with staged fine-tuning and early stopping.

Uses ``tf.GradientTape`` for fine-grained control over differential
learning rates (lower for BERT backbone, higher for MLP heads).
Supports two-stage training: first with frozen BERT, then joint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf

from credit_rating.config.settings import CreditRatingSettings
from credit_rating.models.hybrid import HybridRatingModel
from credit_rating.models.loss import OrdinalCrossEntropyLoss
from credit_rating.training.metrics import TrainingMetrics, compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class TrainingHistory:
    """Record of per-epoch training and validation metrics.

    Args:
        train_losses: Per-epoch training loss.
        val_losses: Per-epoch validation loss.
        val_metrics: Per-epoch validation metrics.
        best_epoch: Epoch with the best validation MAE.
    """

    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_metrics: List[TrainingMetrics] = field(default_factory=list)
    best_epoch: int = 0


class ModelTrainer:
    """Trainer for :class:`HybridRatingModel` with staged fine-tuning.

    Stage 1: Freeze BERT, train structured tower + fusion head.
    Stage 2: Unfreeze BERT, train all components jointly with a
    lower learning rate on BERT parameters.

    Args:
        model: The hybrid model to train.
        settings: Configuration for LR, epochs, patience, etc.
    """

    def __init__(
        self,
        model: HybridRatingModel,
        settings: Optional[CreditRatingSettings] = None,
    ) -> None:
        self._model = model
        self._s = settings or CreditRatingSettings()
        self._loss_fn = OrdinalCrossEntropyLoss(self._s)

    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        checkpoint_dir: Optional[Path] = None,
    ) -> TrainingHistory:
        """Run two-stage training with early stopping.

        Args:
            train_dataset: Training ``tf.data.Dataset``.
            val_dataset: Validation ``tf.data.Dataset``.
            checkpoint_dir: Directory for best-model checkpoints.

        Returns:
            A :class:`TrainingHistory` recording all epochs.
        """
        ckpt_dir = checkpoint_dir or self._s.checkpoint_dir

        logger.info("Stage 1: Training with frozen BERT backbone")
        self._model.freeze_text_tower()
        history_s1 = self._train_loop(
            train_dataset, val_dataset, ckpt_dir,
            learning_rate=self._s.learning_rate,
            max_epochs=self._s.epochs,
        )

        logger.info("Stage 2: Joint fine-tuning with unfrozen BERT")
        self._model.unfreeze_text_tower()
        history_s2 = self._train_loop(
            train_dataset, val_dataset, ckpt_dir,
            learning_rate=self._s.bert_learning_rate,
            max_epochs=self._s.epochs // 2,
        )

        return _merge_histories(history_s1, history_s2)

    def train_structured_only(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        checkpoint_dir: Optional[Path] = None,
    ) -> TrainingHistory:
        """Train using only structured features (no text tower).

        Args:
            train_dataset: Yielding ``(features, labels)`` batches.
            val_dataset: Validation dataset.
            checkpoint_dir: Directory for checkpoints.

        Returns:
            A :class:`TrainingHistory`.
        """
        ckpt_dir = checkpoint_dir or self._s.checkpoint_dir
        return self._train_structured_loop(
            train_dataset, val_dataset, ckpt_dir,
        )

    def _train_loop(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        ckpt_dir: Path,
        learning_rate: float,
        max_epochs: int,
    ) -> TrainingHistory:
        """Inner training loop with early stopping."""
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
        history = TrainingHistory()
        best_mae = float("inf")
        patience_counter = 0

        for epoch in range(max_epochs):
            train_loss = self._run_epoch(train_ds, optimizer, training=True)
            val_loss, val_metrics = self._evaluate(val_ds)

            history.train_losses.append(train_loss)
            history.val_losses.append(val_loss)
            history.val_metrics.append(val_metrics)

            if val_metrics.mae_notches < best_mae:
                best_mae = val_metrics.mae_notches
                history.best_epoch = epoch
                patience_counter = 0
                self._model.save_checkpoint(ckpt_dir)
            else:
                patience_counter += 1

            if patience_counter >= self._s.early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

            if epoch % 10 == 0:
                logger.info(
                    "Epoch %d: loss=%.4f val_loss=%.4f MAE=%.2f",
                    epoch, train_loss, val_loss, val_metrics.mae_notches,
                )

        return history

    def _train_structured_loop(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        ckpt_dir: Path,
    ) -> TrainingHistory:
        """Training loop for structured-only prediction."""
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self._s.learning_rate,
        )
        history = TrainingHistory()
        best_mae = float("inf")
        patience_counter = 0

        for epoch in range(self._s.epochs):
            train_loss = self._run_structured_epoch(
                train_ds, optimizer, training=True,
            )
            val_loss, val_metrics = self._evaluate_structured(val_ds)

            history.train_losses.append(train_loss)
            history.val_losses.append(val_loss)
            history.val_metrics.append(val_metrics)

            if val_metrics.mae_notches < best_mae:
                best_mae = val_metrics.mae_notches
                history.best_epoch = epoch
                patience_counter = 0
                self._model.save_checkpoint(ckpt_dir)
            else:
                patience_counter += 1

            if patience_counter >= self._s.early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        return history

    def _run_epoch(
        self,
        dataset: tf.data.Dataset,
        optimizer: tf.keras.optimizers.Optimizer,
        training: bool,
    ) -> float:
        """Run one epoch over the full-model dataset."""
        total_loss = 0.0
        num_batches = 0
        for batch in dataset:
            inputs, labels = batch
            loss = self._train_step_full(inputs, labels, optimizer, training)
            total_loss += float(loss)
            num_batches += 1
        return total_loss / max(num_batches, 1)

    def _train_step_full(
        self,
        inputs: Dict[str, tf.Tensor],
        labels: tf.Tensor,
        optimizer: tf.keras.optimizers.Optimizer,
        training: bool,
    ) -> tf.Tensor:
        """Single gradient step for the full hybrid model."""
        with tf.GradientTape() as tape:
            logits = self._model(
                inputs["structured"],
                inputs["input_ids"],
                inputs["attention_mask"],
                training=training,
            )
            loss = self._loss_fn(labels, logits)
        if training:
            gradients = tape.gradient(loss, self._model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, self._model.trainable_variables),
            )
        return loss

    def _run_structured_epoch(
        self,
        dataset: tf.data.Dataset,
        optimizer: tf.keras.optimizers.Optimizer,
        training: bool,
    ) -> float:
        """Run one epoch over the structured-only dataset."""
        total_loss = 0.0
        num_batches = 0
        for features_batch, labels_batch in dataset:
            loss = self._train_step_structured(
                features_batch, labels_batch, optimizer, training,
            )
            total_loss += float(loss)
            num_batches += 1
        return total_loss / max(num_batches, 1)

    def _train_step_structured(
        self,
        features: tf.Tensor,
        labels: tf.Tensor,
        optimizer: tf.keras.optimizers.Optimizer,
        training: bool,
    ) -> tf.Tensor:
        """Single gradient step for structured-only prediction."""
        with tf.GradientTape() as tape:
            logits = self._model.predict_from_structured(
                features, training=training,
            )
            loss = self._loss_fn(labels, logits)
        if training:
            gradients = tape.gradient(loss, self._model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, self._model.trainable_variables),
            )
        return loss

    def _evaluate(
        self,
        dataset: tf.data.Dataset,
    ) -> tuple:
        """Evaluate on a full-model validation dataset."""
        all_true, all_pred = [], []
        total_loss = 0.0
        num_batches = 0
        for batch in dataset:
            inputs, labels = batch
            logits = self._model(
                inputs["structured"],
                inputs["input_ids"],
                inputs["attention_mask"],
                training=False,
            )
            loss = self._loss_fn(labels, logits)
            total_loss += float(loss)
            num_batches += 1
            all_true.append(labels.numpy())
            all_pred.append(tf.argmax(logits, axis=-1).numpy())

        avg_loss = total_loss / max(num_batches, 1)
        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        metrics = compute_metrics(y_true, y_pred)
        return avg_loss, metrics

    def _evaluate_structured(
        self,
        dataset: tf.data.Dataset,
    ) -> tuple:
        """Evaluate on a structured-only validation dataset."""
        all_true, all_pred = [], []
        total_loss = 0.0
        num_batches = 0
        for features_batch, labels_batch in dataset:
            logits = self._model.predict_from_structured(
                features_batch, training=False,
            )
            loss = self._loss_fn(labels_batch, logits)
            total_loss += float(loss)
            num_batches += 1
            all_true.append(labels_batch.numpy())
            all_pred.append(tf.argmax(logits, axis=-1).numpy())

        avg_loss = total_loss / max(num_batches, 1)
        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        metrics = compute_metrics(y_true, y_pred)
        return avg_loss, metrics


def _merge_histories(h1: TrainingHistory, h2: TrainingHistory) -> TrainingHistory:
    """Merge two sequential training histories."""
    return TrainingHistory(
        train_losses=h1.train_losses + h2.train_losses,
        val_losses=h1.val_losses + h2.val_losses,
        val_metrics=h1.val_metrics + h2.val_metrics,
        best_epoch=h1.best_epoch
        if h1.val_metrics and h2.val_metrics
        and h1.val_metrics[h1.best_epoch].mae_notches
        <= h2.val_metrics[min(h2.best_epoch, len(h2.val_metrics) - 1)].mae_notches
        else len(h1.val_losses) + h2.best_epoch,
    )
