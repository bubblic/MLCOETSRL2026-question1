"""MLP structured tower for financial ratio features.

Processes the 25-dimensional normalised financial ratio vector through
a multi-layer perceptron and produces a 128-dimensional embedding.
Architecture: ``d -> 256 -> 256 -> 128`` with BatchNormalization and
Dropout between each layer.
"""

from __future__ import annotations

from typing import Optional, Tuple

import tensorflow as tf

from credit_rating.config.settings import CreditRatingSettings


class StructuredTower(tf.keras.Model):
    """MLP tower for structured financial features.

    Args:
        settings: Configuration supplying hidden dimensions and
            dropout rate.
    """

    def __init__(
        self,
        settings: Optional[CreditRatingSettings] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        s = settings or CreditRatingSettings()
        dims = s.structured_hidden_dims
        dropout = s.dropout_rate

        self._layers_list = []
        for dim in dims:
            self._layers_list.append(tf.keras.layers.Dense(dim))
            self._layers_list.append(tf.keras.layers.BatchNormalization())
            self._layers_list.append(tf.keras.layers.ReLU())
            self._layers_list.append(tf.keras.layers.Dropout(dropout))

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Forward pass through the MLP.

        Args:
            inputs: Tensor of shape ``(batch, num_features)``.
            training: Whether to apply dropout and update batch norm.

        Returns:
            Embedding tensor of shape ``(batch, 128)``.
        """
        x = inputs
        for layer in self._layers_list:
            if isinstance(
                layer,
                (tf.keras.layers.Dropout, tf.keras.layers.BatchNormalization),
            ):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x
