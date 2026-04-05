"""Configuration-driven experiment runner for credit rating models.

Provides :class:`CreditRatingExperimentConfig` and
:func:`run_credit_rating_experiment` to eliminate duplication across
entry-point scripts.  Each ``run_*.py`` script becomes a thin config
declaration.

Example::

    from credit_rating.experiment import (
        CreditRatingExperimentConfig,
        run_credit_rating_experiment,
    )

    run_credit_rating_experiment(CreditRatingExperimentConfig())
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import tensorflow as tf  # noqa: F401 — must load before sklearn on Windows
import numpy as np

from credit_rating.config.settings import CreditRatingSettings, RatingClass

logger = logging.getLogger(__name__)


@dataclass
class CreditRatingExperimentConfig:
    """Declares a complete credit-rating training + evaluation experiment.

    Args:
        kaggle_csv_path: Path to the Kaggle Corporate Credit Ratings CSV.
        settings: System-wide configuration.
        checkpoint_dir: Where to save the best model checkpoint.
        output_dir: Where to write evaluation reports and predictions.
        run_training: Whether to train the model.
        run_evaluation: Whether to run backtesting after training.
        run_case_studies: Which case studies to run (empty = none).
        case_study_known_entities: Per-case-study known entities for
            anonymization.
    """

    kaggle_csv_path: Path = Path("data/raw/corporate_ratings.csv")
    settings: CreditRatingSettings = field(default_factory=CreditRatingSettings)
    checkpoint_dir: Path = Path("checkpoints/credit_rating")
    output_dir: Path = Path("outputs/credit_rating")
    run_training: bool = True
    run_evaluation: bool = True
    run_case_studies: List[str] = field(default_factory=lambda: ["evergrande", "enron"])
    case_study_known_entities: Dict[str, List[Dict]] = field(
        default_factory=lambda: {
            "evergrande": [
                {
                    "type": "ORG",
                    "names": [
                        "China Evergrande Group",
                        "Evergrande Group",
                        "Evergrande",
                        "Hengda Real Estate Group",
                    ],
                },
            ],
            "enron": [
                {
                    "type": "ORG",
                    "names": ["Enron Corp.", "Enron"],
                },
            ],
        }
    )


def run_credit_rating_experiment(
    config: CreditRatingExperimentConfig,
) -> Dict:
    """Execute the full credit rating pipeline end-to-end.

    Stages:
        1. Load and prepare the Kaggle dataset
        2. Train the hybrid model (structured tower)
        3. Evaluate via temporal backtesting
        4. Run case studies (Evergrande, Enron)
        5. Save all results

    Args:
        config: Experiment configuration.

    Returns:
        A dict with training metrics, evaluation results, and
        case study outputs.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    results: Dict = {}
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1: Load data
    # ------------------------------------------------------------------
    logger.info("Stage 1: Loading Kaggle dataset from %s", config.kaggle_csv_path)
    features, labels = _load_kaggle_data(config)
    logger.info(
        "Loaded %d samples, %d features, %d classes",
        len(labels), features.shape[1], len(set(labels)),
    )

    # ------------------------------------------------------------------
    # Stage 2: Train
    # ------------------------------------------------------------------
    if config.run_training:
        logger.info("Stage 2: Training hybrid model")
        train_results = _train_model(config, features, labels)
        results["training"] = train_results
    else:
        logger.info("Stage 2: Skipping training")

    # ------------------------------------------------------------------
    # Stage 3: Evaluate
    # ------------------------------------------------------------------
    if config.run_evaluation:
        logger.info("Stage 3: Backtesting")
        eval_results = _evaluate_model(config, features, labels)
        results["evaluation"] = eval_results
    else:
        logger.info("Stage 3: Skipping evaluation")

    # ------------------------------------------------------------------
    # Stage 4: Case studies
    # ------------------------------------------------------------------
    for study_name in config.run_case_studies:
        logger.info("Stage 4: Running case study — %s", study_name)
        study_results = _run_case_study(study_name, config)
        results[f"case_study_{study_name}"] = study_results

    # ------------------------------------------------------------------
    # Stage 5: Save
    # ------------------------------------------------------------------
    output_path = config.output_dir / "experiment_results.json"
    _save_results(results, output_path)
    logger.info("Experiment complete. Results saved to %s", output_path)

    return results


# ------------------------------------------------------------------
# Stage implementations
# ------------------------------------------------------------------


def _load_kaggle_data(
    config: CreditRatingExperimentConfig,
) -> tuple:
    """Load and normalise the Kaggle dataset."""
    from credit_rating.ingestion.kaggle_loader import KaggleRatingsLoader
    from credit_rating.features.normalizer import RatioNormalizer

    loader = KaggleRatingsLoader(
        csv_path=config.kaggle_csv_path,
        settings=config.settings,
    )
    ratios_list = []
    labels_list = []
    for ratios, rating in loader.load():
        ratios_list.append(ratios)
        labels_list.append(rating.value)

    normalizer = RatioNormalizer()
    features = normalizer.fit_transform(ratios_list)
    labels = np.array(labels_list, dtype=np.int32)

    scaler_path = config.checkpoint_dir / "scaler.pkl"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    normalizer.save(scaler_path)
    logger.info("Saved fitted scaler to %s", scaler_path)

    return features, labels


def _train_model(
    config: CreditRatingExperimentConfig,
    features: np.ndarray,
    labels: np.ndarray,
) -> Dict:
    """Train the hybrid model on structured features."""
    from credit_rating.models.hybrid import HybridRatingModel
    from credit_rating.training.dataset import build_rating_dataset, split_dataset
    from credit_rating.training.trainer import ModelTrainer

    (train_f, train_l), (val_f, val_l), (test_f, test_l) = split_dataset(
        features, labels, seed=config.settings.random_seed,
    )

    train_ds = build_rating_dataset(train_f, train_l, config.settings)
    val_ds = build_rating_dataset(val_f, val_l, config.settings, shuffle=False)

    model = HybridRatingModel(settings=config.settings)
    trainer = ModelTrainer(model, settings=config.settings)
    history = trainer.train_structured_only(
        train_ds, val_ds, checkpoint_dir=config.checkpoint_dir,
    )

    return {
        "best_epoch": history.best_epoch,
        "final_train_loss": history.train_losses[-1] if history.train_losses else None,
        "final_val_loss": history.val_losses[-1] if history.val_losses else None,
        "num_epochs": len(history.train_losses),
    }


def _evaluate_model(
    config: CreditRatingExperimentConfig,
    features: np.ndarray,
    labels: np.ndarray,
) -> Dict:
    """Evaluate via ordinal logistic baseline + temporal backtesting."""
    from credit_rating.models.ordinal_logistic import OrdinalLogisticRatingModel
    from credit_rating.training.dataset import split_dataset
    from credit_rating.training.metrics import compute_metrics

    (train_f, train_l), (val_f, val_l), (test_f, test_l) = split_dataset(
        features, labels, seed=config.settings.random_seed,
    )

    baseline = OrdinalLogisticRatingModel(settings=config.settings)
    baseline.fit(train_f, train_l)
    test_pred = baseline.predict_array(test_f)
    metrics = compute_metrics(test_l, test_pred)

    logger.info(
        "Baseline ordinal logistic: accuracy=%.3f, MAE=%.2f, "
        "macro-F1=%.3f, IG accuracy=%.3f",
        metrics.accuracy, metrics.mae_notches,
        metrics.macro_f1, metrics.investment_grade_accuracy,
    )

    return {
        "baseline_accuracy": metrics.accuracy,
        "baseline_mae_notches": metrics.mae_notches,
        "baseline_macro_f1": metrics.macro_f1,
        "baseline_spearman_rho": metrics.spearman_rho,
        "baseline_ig_accuracy": metrics.investment_grade_accuracy,
        "test_samples": len(test_l),
    }


def _run_case_study(
    name: str,
    config: CreditRatingExperimentConfig,
) -> Dict:
    """Run a named case study."""
    if name == "evergrande":
        from credit_rating.case_studies.evergrande import EvergrandeAnalysis
        return EvergrandeAnalysis(settings=config.settings).run()
    if name == "enron":
        from credit_rating.case_studies.enron import EnronAnalysis
        return EnronAnalysis(settings=config.settings).run()
    logger.warning("Unknown case study: %s", name)
    return {"error": f"Unknown case study: {name}"}


def _save_results(results: Dict, path: Path) -> None:
    """Serialise results to JSON, coercing non-serialisable types."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_default)
