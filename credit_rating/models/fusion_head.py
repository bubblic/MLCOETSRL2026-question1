"""Fusion head that merges structured and text embeddings.

Concatenates the 128-dim structured embedding and the 128-dim text
embedding (256 total), passes through a hidden layer, and produces
the final 7-class logits.
"""

from __future__ import annotations

from typing import Optional

import tensorflow as tf

from credit_rating.config.settings import CreditRatingSettings


class FusionHead(tf.keras.Model):
    """Late-fusion classification head.

    Architecture: ``concat(256) -> Dense(256) -> ReLU -> Dropout -> Dense(7)``

    Args:
        settings: Configuration supplying hidden dimension, dropout,
            and number of rating classes.
    """

    def __init__(
        self,
        settings: Optional[CreditRatingSettings] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        s = settings or CreditRatingSettings()

        self._hidden = tf.keras.layers.Dense(
            s.fusion_hidden_dim,
            activation="relu",
        )
        self._dropout = tf.keras.layers.Dropout(s.dropout_rate)
        self._output_layer = tf.keras.layers.Dense(s.num_rating_classes)

    def call(
        self,
        structured_embedding: tf.Tensor,
        text_embedding: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Forward pass: concatenate, hidden layer, output logits.

        Args:
            structured_embedding: Shape ``(batch, structured_dim)``.
            text_embedding: Shape ``(batch, text_dim)``.
            training: Whether to apply dropout.

        Returns:
            Logits tensor of shape ``(batch, num_rating_classes)``.
        """
        concatenated = tf.concat(
            [structured_embedding, text_embedding],
            axis=-1,
        )
        hidden = self._hidden(concatenated)
        dropped = self._dropout(hidden, training=training)
        return self._output_layer(dropped)
