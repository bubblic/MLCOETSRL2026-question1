"""FinBERT text tower with attention pooling.

Wraps a ``TFAutoModel`` (FinBERT by default) and applies learned
attention pooling over token-level embeddings, then projects the
768-dimensional BERT output down to a 128-dimensional embedding.
"""

from __future__ import annotations

import logging
from typing import Optional

import tensorflow as tf

from credit_rating.config.settings import CreditRatingSettings

logger = logging.getLogger(__name__)


class TextTower(tf.keras.Model):
    """FinBERT text tower with attention pooling and projection.

    Args:
        settings: Configuration supplying model name and projection
            dimension.
    """

    def __init__(
        self,
        settings: Optional[CreditRatingSettings] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        s = settings or CreditRatingSettings()

        self._bert = _load_finbert(s.finbert_model_name)
        bert_hidden = 768

        self._attention_weights = tf.keras.layers.Dense(1)
        self._projection = tf.keras.layers.Dense(s.text_projection_dim)
        self._layer_norm = tf.keras.layers.LayerNormalization()
        self._dropout = tf.keras.layers.Dropout(s.dropout_rate)

    def call(
        self,
        input_ids: tf.Tensor,
        attention_mask: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Forward pass: BERT -> attention pool -> project.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)``.
            attention_mask: Mask of shape ``(batch, seq_len)``.
            training: Whether to apply dropout.

        Returns:
            Embedding tensor of shape ``(batch, projection_dim)``.
        """
        bert_output = self._bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            training=training,
        )
        hidden_states = bert_output.last_hidden_state

        pooled = self._attention_pool(hidden_states, attention_mask)
        projected = self._projection(pooled)
        normalised = self._layer_norm(projected)
        return self._dropout(normalised, training=training)

    def _attention_pool(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
    ) -> tf.Tensor:
        """Apply learned attention pooling over token embeddings.

        Args:
            hidden_states: Shape ``(batch, seq_len, hidden)``.
            attention_mask: Shape ``(batch, seq_len)``.

        Returns:
            Pooled tensor of shape ``(batch, hidden)``.
        """
        scores = tf.squeeze(self._attention_weights(hidden_states), axis=-1)
        mask_value = tf.constant(-1e9, dtype=scores.dtype)
        mask_float = tf.cast(attention_mask, scores.dtype)
        scores = tf.where(
            tf.equal(mask_float, 0.0),
            mask_value,
            scores,
        )
        weights = tf.nn.softmax(scores, axis=-1)
        weights = tf.expand_dims(weights, axis=-1)
        return tf.reduce_sum(hidden_states * weights, axis=1)

    def freeze_base(self) -> None:
        """Freeze the BERT backbone weights for staged training."""
        self._bert.trainable = False
        logger.info("FinBERT backbone frozen")

    def unfreeze_base(self) -> None:
        """Unfreeze the BERT backbone for joint fine-tuning."""
        self._bert.trainable = True
        logger.info("FinBERT backbone unfrozen")


def _load_finbert(model_name: str) -> tf.keras.Model:
    """Load FinBERT as a TF model, with a placeholder fallback.

    Args:
        model_name: Hugging Face model identifier.

    Returns:
        A TF model with a ``.last_hidden_state`` output attribute.
    """
    try:
        from transformers import TFAutoModel

        return TFAutoModel.from_pretrained(model_name)
    except (ImportError, OSError):
        logger.warning(
            "Could not load %s; using placeholder embedding layer",
            model_name,
        )
        return _PlaceholderBert()


class _PlaceholderBert(tf.keras.Model):
    """Minimal stand-in when FinBERT is unavailable."""

    def __init__(self) -> None:
        super().__init__()
        self._embedding = tf.keras.layers.Embedding(30522, 768)

    def call(
        self,
        input_ids: tf.Tensor = None,
        attention_mask: tf.Tensor = None,
        training: bool = False,
        **kwargs,
    ) -> "_BertOutput":
        """Return a dummy output matching BERT's interface."""
        embedded = self._embedding(input_ids)
        return _BertOutput(last_hidden_state=embedded)


class _BertOutput:
    """Minimal container mimicking transformers model output."""

    def __init__(self, last_hidden_state: tf.Tensor) -> None:
        self.last_hidden_state = last_hidden_state
