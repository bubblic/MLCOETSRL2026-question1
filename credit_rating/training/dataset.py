"""``tf.data.Dataset`` pipeline for credit rating training.

Builds a ``tf.data.Dataset`` from parallel arrays of features and
labels, with ``.map()``, ``.batch()``, and ``.prefetch()`` applied
for efficient GPU feeding.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from credit_rating.config.settings import CreditRatingSettings

logger = logging.getLogger(__name__)


def build_rating_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    settings: Optional[CreditRatingSettings] = None,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Build a batched, prefetched ``tf.data.Dataset``.

    Args:
        features: 2-D array of shape ``(n_samples, 25)``.
        labels: 1-D array of integer labels.
        settings: Configuration for batch size and seed.
        shuffle: Whether to shuffle the dataset each epoch.

    Returns:
        A ready-to-iterate ``tf.data.Dataset`` yielding
        ``(features_batch, labels_batch)`` tuples.
    """
    s = settings or CreditRatingSettings()
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(features, tf.float32),
            tf.cast(labels, tf.int32),
        ),
    )
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(features),
            seed=s.random_seed,
        )
    dataset = dataset.batch(s.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def build_text_rating_dataset(
    structured_features: np.ndarray,
    input_ids: np.ndarray,
    attention_masks: np.ndarray,
    labels: np.ndarray,
    settings: Optional[CreditRatingSettings] = None,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Build a dataset with both structured features and tokenised text.

    Args:
        structured_features: Shape ``(n, 25)`` ratio features.
        input_ids: Shape ``(n, seq_len)`` token IDs.
        attention_masks: Shape ``(n, seq_len)`` attention masks.
        labels: Shape ``(n,)`` integer labels.
        settings: Configuration for batch size and seed.
        shuffle: Whether to shuffle each epoch.

    Returns:
        A ``tf.data.Dataset`` yielding
        ``((struct, ids, mask), labels)`` tuples.
    """
    s = settings or CreditRatingSettings()
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                "structured": tf.cast(structured_features, tf.float32),
                "input_ids": tf.cast(input_ids, tf.int32),
                "attention_mask": tf.cast(attention_masks, tf.int32),
            },
            tf.cast(labels, tf.int32),
        ),
    )
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(labels),
            seed=s.random_seed,
        )
    dataset = dataset.batch(s.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def split_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Split data into train / validation / test sets.

    Args:
        features: Feature matrix.
        labels: Label vector.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        seed: Random seed for reproducibility.

    Returns:
        Three ``(features, labels)`` tuples for train, val, and test.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(labels))
    n_train = int(len(labels) * train_ratio)
    n_val = int(len(labels) * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return (
        (features[train_idx], labels[train_idx]),
        (features[val_idx], labels[val_idx]),
        (features[test_idx], labels[test_idx]),
    )
