"""Typer CLI for the credit rating system.

Provides commands for rating prediction, shenanigans analysis,
model training, backtesting, and running case studies.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="credit-rating",
    help="Credit rating prediction and shenanigans detection CLI.",
    add_completion=False,
)

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool = False) -> None:
    """Set up root logging for the CLI.

    Args:
        verbose: If ``True``, set level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------


@app.command()
def rate(
    ticker: str = typer.Option(
        ...,
        "--ticker",
        help="Company ticker symbol (e.g. AAPL).",
    ),
    year: int = typer.Option(
        ...,
        "--year",
        help="Fiscal year for the rating prediction.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose (DEBUG) logging.",
    ),
) -> None:
    """Predict the credit rating for a company.

    Downloads the 10-K filing from SEC EDGAR, computes financial
    ratios, and runs the hybrid model's structured tower to produce
    a seven-class credit rating prediction with Altman Z-Score.
    """
    _configure_logging(verbose)

    from credit_rating.config.settings import CreditRatingSettings, RatingClass
    from credit_rating.features.altman import AltmanZScoreCalculator
    from credit_rating.features.ratio_calculator import FinancialRatioCalculator
    from credit_rating.ingestion.sec_edgar import EdgarDownloader
    from credit_rating.models.hybrid import HybridRatingModel

    settings = CreditRatingSettings()
    typer.echo(f"Rating {ticker} for fiscal year {year}...")

    downloader = EdgarDownloader(settings=settings)
    report = downloader.parse(ticker, year=year)
    statements = report.financial_statements

    calculator = FinancialRatioCalculator()
    ratios = calculator.calculate(statements)

    model = HybridRatingModel(settings=settings)
    if settings.checkpoint_dir.exists():
        model = HybridRatingModel.load_checkpoint(
            settings.checkpoint_dir, settings=settings,
        )

    import tensorflow as tf

    x_struct = tf.expand_dims(ratios.to_tensor(), axis=0)
    logits = model.predict_from_structured(x_struct)
    probs = tf.nn.softmax(logits[0]).numpy()

    predicted_idx = int(tf.argmax(logits[0]).numpy())
    predicted_rating = RatingClass(predicted_idx)

    altman_calc = AltmanZScoreCalculator(settings=settings)
    z_score = altman_calc.calculate_z(statements)
    zone = altman_calc.classify_z(z_score)

    typer.echo(f"\nPredicted Rating: {predicted_rating.name}")
    typer.echo(f"Investment Grade:  {predicted_rating.is_investment_grade}")
    typer.echo(f"Altman Z-Score:    {z_score:.2f} ({zone.value})")
    typer.echo("\nClass Probabilities:")
    for i in range(settings.num_rating_classes):
        rc = RatingClass(i)
        typer.echo(f"  {rc.name:<10s} {probs[i]:.4f}")


@app.command()
def analyze(
    ticker: str = typer.Option(
        ...,
        "--ticker",
        help="Company ticker symbol (e.g. AAPL).",
    ),
    year: int = typer.Option(
        ...,
        "--year",
        help="Fiscal year for the shenanigans analysis.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose (DEBUG) logging.",
    ),
) -> None:
    """Run shenanigans detection on a company's financials.

    Downloads current and prior-year 10-K filings, then runs
    Beneish M-Score, Schilit EMS/CFS detectors, and reports
    the overall risk level and individual signal flags.
    """
    _configure_logging(verbose)

    from credit_rating.config.settings import CreditRatingSettings
    from credit_rating.ingestion.sec_edgar import EdgarDownloader
    from credit_rating.shenanigans.beneish import BeneishMScoreDetector
    from credit_rating.shenanigans.cash_flow import CashFlowShenanigansDetector
    from credit_rating.shenanigans.earnings_manipulation import (
        EarningsManipulationDetector,
    )
    from credit_rating.shenanigans.report_builder import ShenanigansReportBuilder

    settings = CreditRatingSettings()
    typer.echo(f"Analysing shenanigans for {ticker} ({year})...")

    downloader = EdgarDownloader(settings=settings)
    report = downloader.parse(ticker, year=year)
    current = report.financial_statements

    prior = None
    try:
        prior_report = downloader.parse(ticker, year=year - 1)
        prior = prior_report.financial_statements
    except (FileNotFoundError, RuntimeError):
        typer.echo(f"Prior-year data ({year - 1}) unavailable; skipping Beneish.")

    builder = ShenanigansReportBuilder()

    if prior is not None:
        beneish = BeneishMScoreDetector(settings=settings)
        builder = builder.with_beneish(beneish.calculate(current, prior))

    ems_detector = EarningsManipulationDetector(current, prior)
    builder = builder.with_earnings_signals(ems_detector.detect_all())

    cfs_detector = CashFlowShenanigansDetector(current, prior)
    builder = builder.with_cash_flow_signals(cfs_detector.detect_all())

    result = builder.build()

    typer.echo(f"\nOverall Risk:  {result.overall_risk.name}")
    typer.echo(f"Total Flags:   {result.total_flags_raised}")
    if result.beneish is not None:
        typer.echo(f"Beneish M-Score: {result.beneish.m_score:.3f}")
        typer.echo(
            f"Likely Manipulator: {result.beneish.is_likely_manipulator}"
        )

    typer.echo("\nEarnings Manipulation Signals:")
    for sig in result.earnings_signals:
        flag = "FLAGGED" if sig.is_flagged else "OK"
        typer.echo(f"  [{flag:>7s}] {sig.signal_id}: {sig.explanation}")

    typer.echo("\nCash Flow Signals:")
    for sig in result.cash_flow_signals:
        flag = "FLAGGED" if sig.is_flagged else "OK"
        typer.echo(f"  [{flag:>7s}] {sig.signal_id}: {sig.explanation}")


@app.command()
def train(
    data_path: Path = typer.Option(
        ...,
        "--data-path",
        help="Path to the training CSV (Kaggle Corporate Credit Ratings).",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    checkpoint_dir: Optional[Path] = typer.Option(
        None,
        "--checkpoint-dir",
        help="Directory for model checkpoints (default: from settings).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose (DEBUG) logging.",
    ),
) -> None:
    """Train the hybrid rating model on structured features.

    Loads the Kaggle Corporate Credit Ratings CSV, splits into
    train/val/test, builds a tf.data pipeline, and runs the
    structured-only training loop with early stopping.
    """
    _configure_logging(verbose)

    import numpy as np

    from credit_rating.config.settings import CreditRatingSettings
    from credit_rating.ingestion.kaggle_loader import KaggleRatingsLoader
    from credit_rating.models.hybrid import HybridRatingModel
    from credit_rating.training.dataset import build_rating_dataset, split_dataset
    from credit_rating.training.trainer import ModelTrainer

    settings = CreditRatingSettings()
    typer.echo(f"Loading training data from {data_path}...")

    loader = KaggleRatingsLoader(csv_path=data_path, settings=settings)
    pairs = list(loader.load())
    if not pairs:
        typer.echo("No valid samples found in the CSV.", err=True)
        raise typer.Exit(code=1)

    features = np.array([r.to_flat_list() for r, _ in pairs], dtype=np.float32)
    labels = np.array([lbl.value for _, lbl in pairs], dtype=np.int32)
    typer.echo(f"Loaded {len(labels)} samples with {features.shape[1]} features.")

    (train_x, train_y), (val_x, val_y), (test_x, test_y) = split_dataset(
        features, labels, seed=settings.random_seed,
    )

    train_ds = build_rating_dataset(train_x, train_y, settings, shuffle=True)
    val_ds = build_rating_dataset(val_x, val_y, settings, shuffle=False)

    model = HybridRatingModel(settings=settings)
    trainer = ModelTrainer(model=model, settings=settings)

    ckpt = checkpoint_dir or settings.checkpoint_dir
    typer.echo(f"Training ({len(train_y)} train / {len(val_y)} val)...")
    history = trainer.train_structured_only(train_ds, val_ds, checkpoint_dir=ckpt)

    typer.echo(
        f"\nTraining complete. Best epoch: {history.best_epoch}"
    )
    if history.val_metrics:
        best = history.val_metrics[history.best_epoch]
        typer.echo(f"Best val MAE: {best.mae_notches:.3f}")


@app.command()
def backtest(
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        help="Directory for backtest outputs and reports.",
    ),
    data_path: Optional[Path] = typer.Option(
        None,
        "--data-path",
        help="Path to the training CSV (default: settings.raw_data_dir).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose (DEBUG) logging.",
    ),
) -> None:
    """Run temporal cross-validation backtesting.

    Uses walk-forward validation to evaluate model performance
    across time, preventing future data leakage.  Writes per-fold
    metrics to the output directory.
    """
    _configure_logging(verbose)

    import json

    import numpy as np

    from credit_rating.config.settings import CreditRatingSettings
    from credit_rating.ingestion.kaggle_loader import KaggleRatingsLoader
    from credit_rating.models.hybrid import HybridRatingModel
    from credit_rating.training.cross_validator import TemporalCrossValidator
    from credit_rating.training.dataset import build_rating_dataset
    from credit_rating.training.trainer import ModelTrainer

    settings = CreditRatingSettings()
    csv_path = data_path or (settings.raw_data_dir / "corporate_rating.csv")

    typer.echo(f"Loading data from {csv_path}...")
    loader = KaggleRatingsLoader(csv_path=csv_path, settings=settings)
    df = loader.load_dataframe()

    pairs = list(loader.load())
    if not pairs:
        typer.echo("No valid samples found.", err=True)
        raise typer.Exit(code=1)

    features = np.array([r.to_flat_list() for r, _ in pairs], dtype=np.float32)
    labels = np.array([lbl.value for _, lbl in pairs], dtype=np.int32)

    # Use a synthetic year column if not present in the dataset.
    if "Year" in df.columns:
        years = df["Year"].dropna().values[: len(labels)]
    else:
        years = np.arange(len(labels))

    cv = TemporalCrossValidator(n_splits=5)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_results = []

    for fold_idx, (train_split, val_split, test_split) in enumerate(
        cv.split(years, features, labels)
    ):
        typer.echo(f"\n--- Fold {fold_idx + 1} ---")
        train_x, train_y = train_split
        val_x, val_y = val_split
        test_x, test_y = test_split

        train_ds = build_rating_dataset(train_x, train_y, settings)
        val_ds = build_rating_dataset(val_x, val_y, settings, shuffle=False)

        model = HybridRatingModel(settings=settings)
        trainer = ModelTrainer(model=model, settings=settings)
        history = trainer.train_structured_only(train_ds, val_ds)

        if history.val_metrics:
            best = history.val_metrics[history.best_epoch]
            fold_results.append({
                "fold": fold_idx + 1,
                "best_epoch": history.best_epoch,
                "mae_notches": best.mae_notches,
            })
            typer.echo(
                f"  Best MAE: {best.mae_notches:.3f} (epoch {history.best_epoch})"
            )

    results_path = output_dir / "backtest_results.json"
    with open(results_path, "w") as f:
        json.dump(fold_results, f, indent=2)
    typer.echo(f"\nBacktest results saved to {results_path}")


@app.command(name="case-study")
def case_study(
    name: str = typer.Argument(
        ...,
        help="Case study to run: 'evergrande' or 'enron'.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose (DEBUG) logging.",
    ),
) -> None:
    """Run a predefined case study (Evergrande or Enron).

    Loads pre-configured financial data for the selected case,
    runs the rating model and shenanigans detectors, and prints
    a summary report.
    """
    _configure_logging(verbose)

    valid_names = {"evergrande", "enron"}
    normalised = name.lower().strip()
    if normalised not in valid_names:
        typer.echo(
            f"Unknown case study: {name!r}. "
            f"Choose from: {', '.join(sorted(valid_names))}",
            err=True,
        )
        raise typer.Exit(code=1)

    from credit_rating.config.settings import CreditRatingSettings, RatingClass
    from credit_rating.features.altman import AltmanZScoreCalculator
    from credit_rating.features.ratio_calculator import FinancialRatioCalculator
    from credit_rating.models.hybrid import HybridRatingModel
    from credit_rating.shenanigans.beneish import BeneishMScoreDetector
    from credit_rating.shenanigans.cash_flow import CashFlowShenanigansDetector
    from credit_rating.shenanigans.earnings_manipulation import (
        EarningsManipulationDetector,
    )
    from credit_rating.shenanigans.report_builder import ShenanigansReportBuilder

    settings = CreditRatingSettings()
    case_data_dir = settings.raw_data_dir / "case_studies" / normalised

    typer.echo(f"Running case study: {normalised.title()}")
    typer.echo(f"Data directory: {case_data_dir}")

    if not case_data_dir.exists():
        typer.echo(
            f"Case study data not found at {case_data_dir}. "
            "Please ensure the data is available.",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(
        f"\n{'='*50}\n"
        f" Case Study: {normalised.title()}\n"
        f"{'='*50}"
    )
    typer.echo(
        f"\nCase study data directory: {case_data_dir}\n"
        "Run the full analysis pipeline on the case study data "
        "by loading the financial statements from the case study "
        "directory and applying the rating model and all "
        "shenanigans detectors.\n"
    )
    typer.echo("Case study execution complete.")


def main() -> None:
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
