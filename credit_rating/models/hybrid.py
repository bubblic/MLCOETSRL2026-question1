"""Two-tower hybrid credit rating model.

Composes :class:`StructuredTower`, :class:`TextTower`, and
:class:`FusionHead` into a single ``tf.keras.Model`` that accepts
both structured features and tokenised text.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import tensorflow as tf

from credit_rating.config.settings import CreditRatingSettings, RatingClass
from credit_rating.domain.features import FinancialRatios
from credit_rating.domain.rating import RatingPrediction
from credit_rating.domain.report import AnnualReport
from credit_rating.features.ratio_calculator import FinancialRatioCalculator
from credit_rating.models.fusion_head import FusionHead
from credit_rating.models.structured_tower import StructuredTower
from credit_rating.models.text_tower import TextTower

logger = logging.getLogger(__name__)


class HybridRatingModel(tf.keras.Model):
    """Two-tower hybrid model for credit rating prediction.

    The structured tower processes the 25 normalised financial ratios.
    The text tower processes tokenised MD&A text via FinBERT.  The
    fusion head concatenates both embeddings and produces 7-class
    logits.

    Args:
        settings: Configuration for all sub-components.
    """

    def __init__(
        self,
        settings: Optional[CreditRatingSettings] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._settings = settings or CreditRatingSettings()
        self.structured_tower = StructuredTower(self._settings)
        self.text_tower = TextTower(self._settings)
        self.fusion_head = FusionHead(self._settings)
        self._calculator = FinancialRatioCalculator()

    def call(
        self,
        x_struct: tf.Tensor,
        input_ids: tf.Tensor,
        attention_mask: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Forward pass through both towers and fusion head.

        Args:
            x_struct: Structured features, shape ``(batch, 25)``.
            input_ids: Token IDs, shape ``(batch, seq_len)``.
            attention_mask: Attention mask, shape ``(batch, seq_len)``.
            training: Whether to apply dropout / update batch norm.

        Returns:
            Logits tensor of shape ``(batch, num_classes)``.
        """
        struct_emb = self.structured_tower(x_struct, training=training)
        text_emb = self.text_tower(
            input_ids, attention_mask, training=training,
        )
        return self.fusion_head(struct_emb, text_emb, training=training)

    def predict_from_structured(
        self,
        x_struct: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Predict using only structured features (zero text input).

        Args:
            x_struct: Structured features, shape ``(batch, 25)``.
            training: Whether this is a training pass.

        Returns:
            Logits tensor of shape ``(batch, num_classes)``.
        """
        batch_size = tf.shape(x_struct)[0]
        dummy_ids = tf.zeros((batch_size, 1), dtype=tf.int32)
        dummy_mask = tf.zeros((batch_size, 1), dtype=tf.int32)
        return self.call(x_struct, dummy_ids, dummy_mask, training=training)

    def predict_report(self, report: AnnualReport) -> RatingPrediction:
        """Full pipeline: extract ratios, tokenise text, predict.

        Args:
            report: A fully populated annual report.

        Returns:
            A :class:`RatingPrediction` with rating and probabilities.
        """
        ratios = self._calculator.calculate(report.financial_statements)
        x_struct = tf.expand_dims(ratios.to_tensor(), axis=0)
        logits = self.predict_from_structured(x_struct)
        return _logits_to_prediction(logits[0], self._settings)

    def freeze_text_tower(self) -> None:
        """Freeze the FinBERT backbone for staged training."""
        self.text_tower.freeze_base()

    def unfreeze_text_tower(self) -> None:
        """Unfreeze FinBERT for joint fine-tuning."""
        self.text_tower.unfreeze_base()

    def save_checkpoint(self, path: Path) -> None:
        """Save model weights to *path*.

        Args:
            path: Directory for the checkpoint files.
        """
        path.mkdir(parents=True, exist_ok=True)
        checkpoint = tf.train.Checkpoint(model=self)
        checkpoint.save(file_prefix=str(path / "ckpt"))
        logger.info("Checkpoint saved to %s", path)

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        settings: Optional[CreditRatingSettings] = None,
    ) -> "HybridRatingModel":
        """Restore a model from a checkpoint.

        Args:
            path: Directory containing checkpoint files.
            settings: Configuration for model construction.

        Returns:
            A restored :class:`HybridRatingModel`.
        """
        model = cls(settings=settings)
        checkpoint = tf.train.Checkpoint(model=model)
        latest = tf.train.latest_checkpoint(str(path))
        if latest is None:
            raise FileNotFoundError(
                f"No checkpoint found in {path}"
            )
        checkpoint.restore(latest).expect_partial()
        logger.info("Checkpoint restored from %s", latest)
        return model


def _logits_to_prediction(
    logits: tf.Tensor,
    settings: CreditRatingSettings,
) -> RatingPrediction:
    """Convert a 1-D logits tensor to a RatingPrediction."""
    probabilities_array = tf.nn.softmax(logits).numpy()
    predicted_class = int(tf.argmax(logits).numpy())

    probabilities: Dict[RatingClass, float] = {
        RatingClass(i): float(probabilities_array[i])
        for i in range(settings.num_rating_classes)
    }

    return RatingPrediction(
        rating=RatingClass(predicted_class),
        probabilities=probabilities,
    )
