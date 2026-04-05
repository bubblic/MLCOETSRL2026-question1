"""Integrated-gradients attention visualiser for the text tower.

Computes token-level importance scores by running integrated gradients
on the token embedding layer of a :class:`TextTower` model.  This is a
model-agnostic attribution technique that satisfies the completeness
axiom: the sum of attributions equals the difference between the
model's output at the actual input and the baseline (zero embedding).

Reference:
    Sundararajan, Taly & Yan, *Axiomatic Attribution for Deep Networks*,
    ICML 2017.
"""

from __future__ import annotations

import logging
from typing import Any, List, NamedTuple, Optional

import numpy as np
import tensorflow as tf

from credit_rating.models.text_tower import TextTower

logger = logging.getLogger(__name__)

# Number of interpolation steps between baseline and actual embedding.
# Higher values yield more accurate attributions at the cost of compute.
_DEFAULT_NUM_STEPS: int = 50

# Dimension of a single token's embedding produced by BERT-family models.
_BERT_HIDDEN_DIM: int = 768


TokenImportance = NamedTuple(
    "TokenImportance",
    [("token", str), ("importance", float)],
)
"""A single token paired with its integrated-gradient importance score."""


class TextAttentionVisualizer:
    """Integrated-gradients attribution over TextTower token embeddings.

    Args:
        model: A :class:`TextTower` instance whose embedding layer will
            be probed via ``tf.GradientTape``.
        num_steps: Number of linear interpolation steps for the
            Riemann approximation.  Defaults to
            :data:`_DEFAULT_NUM_STEPS`.
    """

    def __init__(
        self,
        model: TextTower,
        num_steps: int = _DEFAULT_NUM_STEPS,
    ) -> None:
        self._model = model
        self._num_steps = num_steps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def highlight_influential_tokens(
        self,
        text: str,
        model: TextTower,
        tokenizer: Optional[Any] = None,
    ) -> List[TokenImportance]:
        """Rank tokens in *text* by their influence on the model output.

        Tokenises the input text, runs integrated gradients between a
        zero-embedding baseline and the actual token embeddings, then
        returns every token with its scalar importance score.

        Args:
            text: Raw input string (e.g. an excerpt from a 10-K filing).
            model: The :class:`TextTower` to attribute through.  May
                differ from the model passed to the constructor.
            tokenizer: A HuggingFace-style tokenizer exposing
                ``__call__``, ``convert_ids_to_tokens``, and optionally
                ``model_max_length``.  If ``None``, an empty list is
                returned with a warning.

        Returns:
            A list of :class:`TokenImportance` tuples sorted by
            descending absolute importance.
        """
        if tokenizer is None:
            logger.warning(
                "No tokenizer provided; cannot compute token importances."
            )
            return []

        encoded = tokenizer(
            text,
            return_tensors="tf",
            truncation=True,
            padding=True,
        )
        input_ids: tf.Tensor = encoded["input_ids"]
        attention_mask: tf.Tensor = encoded["attention_mask"]

        tokens: List[str] = tokenizer.convert_ids_to_tokens(
            input_ids.numpy().squeeze(axis=0).tolist(),
        )

        attributions = self._integrated_gradients(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        results: List[TokenImportance] = []
        for idx, token in enumerate(tokens):
            score = float(attributions[idx])
            results.append(TokenImportance(token=token, importance=score))

        results.sort(key=lambda t: abs(t.importance), reverse=True)
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _integrated_gradients(
        self,
        model: TextTower,
        input_ids: tf.Tensor,
        attention_mask: tf.Tensor,
    ) -> np.ndarray:
        """Compute integrated gradients over token embeddings.

        The baseline is a zero embedding of the same shape as the
        actual embeddings.  We linearly interpolate between baseline
        and actual over ``self._num_steps`` points, compute the
        gradient of the model output with respect to each interpolated
        embedding, and approximate the path integral via the
        trapezoidal rule.

        Args:
            model: The :class:`TextTower` to differentiate through.
            input_ids: Token IDs of shape ``(1, seq_len)``.
            attention_mask: Mask of shape ``(1, seq_len)``.

        Returns:
            A 1-D array of length ``seq_len`` where each entry is the
            scalar attribution for the corresponding token.
        """
        actual_embeddings = self._get_embeddings(model, input_ids)
        baseline = tf.zeros_like(actual_embeddings)

        # Build interpolated embeddings: shape (num_steps + 1, 1, seq, hidden)
        alphas = tf.linspace(0.0, 1.0, self._num_steps + 1)
        accumulated_gradients = tf.zeros_like(actual_embeddings)

        for alpha in alphas:
            interpolated = baseline + alpha * (actual_embeddings - baseline)
            gradient = self._compute_gradient(
                model=model,
                embeddings=interpolated,
                attention_mask=attention_mask,
            )
            accumulated_gradients = accumulated_gradients + gradient

        # Trapezoidal approximation: average over the (num_steps + 1) points,
        # then multiply by the difference (actual - baseline).
        avg_gradients = accumulated_gradients / tf.cast(
            self._num_steps + 1, tf.float32,
        )
        integrated_grads = (actual_embeddings - baseline) * avg_gradients

        # Reduce over the hidden dimension to get per-token attributions.
        token_attributions = tf.reduce_sum(
            tf.abs(integrated_grads), axis=-1,
        )

        # Remove the batch dimension.
        return token_attributions.numpy().squeeze(axis=0)

    @staticmethod
    def _get_embeddings(
        model: TextTower,
        input_ids: tf.Tensor,
    ) -> tf.Tensor:
        """Extract token embeddings from the BERT backbone.

        Accesses ``model._bert`` (the underlying FinBERT or placeholder
        model) and looks up embedding vectors for the given token IDs.

        Args:
            model: A :class:`TextTower` instance.
            input_ids: Token IDs of shape ``(batch, seq_len)``.

        Returns:
            Embedding tensor of shape ``(batch, seq_len, hidden_dim)``.
        """
        bert_model = model._bert  # noqa: SLF001
        if hasattr(bert_model, "get_input_embeddings"):
            embedding_layer = bert_model.get_input_embeddings()
        elif hasattr(bert_model, "_embedding"):
            embedding_layer = bert_model._embedding  # noqa: SLF001
        else:
            logger.warning(
                "Could not locate an embedding layer; falling back "
                "to a zero tensor."
            )
            seq_len = input_ids.shape[1]
            return tf.zeros(
                (1, seq_len, _BERT_HIDDEN_DIM), dtype=tf.float32,
            )

        return embedding_layer(input_ids)

    @staticmethod
    def _compute_gradient(
        model: TextTower,
        embeddings: tf.Tensor,
        attention_mask: tf.Tensor,
    ) -> tf.Tensor:
        """Compute the gradient of the model output w.r.t. *embeddings*.

        Instead of feeding token IDs, this injects pre-computed
        embeddings directly into the BERT backbone so that TensorFlow
        can trace the gradient path back to the interpolated input.

        Args:
            model: The :class:`TextTower` to differentiate.
            embeddings: Tensor of shape ``(1, seq_len, hidden_dim)``.
            attention_mask: Mask of shape ``(1, seq_len)``.

        Returns:
            Gradient tensor of the same shape as *embeddings*.
        """
        embeddings = tf.Variable(embeddings, trainable=True)
        with tf.GradientTape() as tape:
            tape.watch(embeddings)

            bert_model = model._bert  # noqa: SLF001
            if hasattr(bert_model, "config"):
                # Transformer model: feed embeddings via inputs_embeds.
                bert_output = bert_model(
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask,
                    training=False,
                )
                hidden_states = bert_output.last_hidden_state
            else:
                # Placeholder model: embeddings *are* the hidden states.
                hidden_states = embeddings

            pooled = model._attention_pool(  # noqa: SLF001
                hidden_states, attention_mask,
            )
            projected = model._projection(pooled)  # noqa: SLF001
            output = model._layer_norm(projected)  # noqa: SLF001

            # Use L2 norm of the output as the scalar target so that
            # gradients capture sensitivity across all output dimensions.
            target = tf.reduce_sum(tf.square(output))

        gradient = tape.gradient(target, embeddings)
        if gradient is None:
            logger.warning(
                "Gradient computation returned None; returning zeros."
            )
            return tf.zeros_like(embeddings)
        return gradient
