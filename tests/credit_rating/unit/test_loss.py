"""Tests for OrdinalCrossEntropyLoss."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from credit_rating.models.loss import OrdinalCrossEntropyLoss


class TestOrdinalCrossEntropyLoss:
    """Verify loss computation and gradient properties."""

    @pytest.fixture
    def loss_fn(self):
        return OrdinalCrossEntropyLoss()

    def test_loss_is_scalar(self, loss_fn):
        logits = tf.random.normal((8, 7))
        labels = tf.constant([0, 1, 2, 3, 4, 5, 6, 0])
        loss = loss_fn(labels, logits)
        assert loss.shape == ()

    def test_loss_is_positive(self, loss_fn):
        logits = tf.random.normal((8, 7))
        labels = tf.constant([0, 1, 2, 3, 4, 5, 6, 0])
        loss = loss_fn(labels, logits)
        assert float(loss) > 0

    def test_perfect_prediction_low_loss(self, loss_fn):
        """Perfect logits should yield near-zero loss."""
        labels = tf.constant([0, 1, 2])
        logits = tf.constant([
            [10.0, -10, -10, -10, -10, -10, -10],
            [-10, 10.0, -10, -10, -10, -10, -10],
            [-10, -10, 10.0, -10, -10, -10, -10],
        ])
        loss = float(loss_fn(labels, logits))
        assert loss < 0.1

    def test_distance_penalty_increases_for_far_predictions(self, loss_fn):
        """Predicting D (6) for AAA (0) should incur higher penalty."""
        labels = tf.constant([0])
        close_logits = tf.constant([[-10, 10.0, -10, -10, -10, -10, -10]])
        far_logits = tf.constant([[-10, -10, -10, -10, -10, -10, 10.0]])
        loss_close = float(loss_fn(labels, close_logits))
        loss_far = float(loss_fn(labels, far_logits))
        assert loss_far > loss_close

    def test_gradients_flow(self, loss_fn):
        """Verify gradients are non-None via tf.GradientTape."""
        logits = tf.Variable(tf.random.normal((4, 7)))
        labels = tf.constant([0, 2, 4, 6])
        with tf.GradientTape() as tape:
            loss = loss_fn(labels, logits)
        grads = tape.gradient(loss, logits)
        assert grads is not None
        assert not tf.reduce_all(tf.equal(grads, 0.0))
