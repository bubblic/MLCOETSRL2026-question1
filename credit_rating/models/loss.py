"""Ordinal cross-entropy loss with distance penalty.

Standard cross-entropy treats all misclassifications equally.  This
loss adds a penalty proportional to the ordinal distance between
the predicted and true rating class, encouraging the model to make
"near misses" rather than large errors.

``L = CE(y, y_hat) + lambda * sum_k[ |k - y_true| * p_k ]``
"""

from __future__ import annotations

from typing import Optional

import tensorflow as tf

from credit_rating.config.settings import CreditRatingSettings


class OrdinalCrossEntropyLoss(tf.keras.losses.Loss):
    """Cross-entropy with ordinal distance penalty.

    Args:
        settings: Configuration supplying ``lambda_ordinal`` and
            ``num_rating_classes``.
    """

    def __init__(
        self,
        settings: Optional[CreditRatingSettings] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        s = settings or CreditRatingSettings()
        self._lambda = s.lambda_ordinal
        self._num_classes = s.num_rating_classes

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """Compute the combined loss.

        Args:
            y_true: Integer labels of shape ``(batch,)``.
            y_pred: Logits of shape ``(batch, num_classes)``.

        Returns:
            Scalar loss tensor.
        """
        ce_loss = self._cross_entropy(y_true, y_pred)
        ordinal_penalty = self._distance_penalty(y_true, y_pred)
        return ce_loss + self._lambda * ordinal_penalty

    @staticmethod
    def _cross_entropy(
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """Standard sparse categorical cross-entropy from logits."""
        return tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                y_true, y_pred, from_logits=True,
            )
        )

    def _distance_penalty(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """Compute the ordinal distance penalty term.

        For each sample, weight each class probability by its distance
        from the true class, then average over the batch.
        """
        probabilities = tf.nn.softmax(y_pred, axis=-1)
        class_indices = tf.cast(
            tf.range(self._num_classes),
            dtype=tf.float32,
        )
        y_true_float = tf.cast(y_true, dtype=tf.float32)
        y_true_expanded = tf.expand_dims(y_true_float, axis=-1)
        distances = tf.abs(class_indices - y_true_expanded)
        weighted = probabilities * distances
        return tf.reduce_mean(tf.reduce_sum(weighted, axis=-1))
