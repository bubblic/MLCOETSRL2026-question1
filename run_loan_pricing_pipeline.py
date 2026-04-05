"""Run the full term loan pricing pipeline end-to-end.

Demonstrates the complete flow from data ingestion through model
training to pricing a single loan, using synthetic data so the script
is self-contained and runnable without API keys.

Steps:
    1. Generate synthetic loan-level dataset
    2. Feature engineering (z-score normalisation, interaction terms)
    3. Train PD classifier (TF neural network, stratified CV)
    4. Train spread regressor (TF neural network, PD-augmented features)
    5. Train quantile regressors + conformal calibration
    6. Calibrate OU process on synthetic spread series
    7. Run Monte Carlo simulation
    8. Price a single example loan

Outputs:
    - Model artefacts saved to ``loan_pricing/artifacts/processed/models/``
    - Evaluation metrics printed to stdout

Usage::

    python run_loan_pricing_pipeline.py
"""

import numpy as np
import tensorflow as tf

from loan_pricing.config import ProjectConfig
from loan_pricing.features.engineering import FeatureConfig, FeatureEngineer
from loan_pricing.logging_config import get_logger
from loan_pricing.models.ou_calibration import (
    MonteCarloLoanPricer,
    OrnsteinUhlenbeckCalibrator,
    VasicekParameters,
)
from loan_pricing.models.pd_model import (
    PDModelConfig,
    TFPDClassifier,
    evaluate_pd_model,
)
from loan_pricing.models.quantile_model import (
    ConformalPredictionInterval,
    QuantileModelConfig,
    TFQuantileRegressor,
)
from loan_pricing.models.spread_model import (
    LoanPricer,
    LoanPricingInput,
    SpreadModelConfig,
    TFSpreadRegressor,
    evaluate_spread_model,
)

logger = get_logger("loan_pricing.pipeline")


if __name__ == "__main__":

    config = ProjectConfig()
    tf.random.set_seed(config.random_seed)
    rng = np.random.default_rng(config.random_seed)

    # -- Step 1: Synthetic dataset ------------------------------------------
    logger.info("Generating synthetic loan dataset")

    n_samples = 500
    n_features = 8
    X_pos = rng.normal(loc=1.0, scale=1.0, size=(n_samples // 2, n_features))
    X_neg = rng.normal(loc=-1.0, scale=1.0, size=(n_samples // 2, n_features))
    X = np.vstack([X_pos, X_neg])
    y_default = np.array(
        [1] * (n_samples // 2) + [0] * (n_samples // 2), dtype=np.float64,
    )
    y_spread = 200.0 + 50.0 * X[:, 0] + 30.0 * y_default + rng.normal(0, 15, n_samples)
    y_spread = np.abs(y_spread)

    idx = rng.permutation(n_samples)
    X, y_default, y_spread = X[idx], y_default[idx], y_spread[idx]

    # Time-aware split.
    train_end = int(n_samples * 0.70)
    val_end = int(n_samples * 0.85)
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_def_train, y_def_test = y_default[:train_end], y_default[val_end:]
    y_spr_train, y_spr_test = y_spread[:train_end], y_spread[val_end:]
    X_cal, y_spr_cal = X[train_end:val_end], y_spread[train_end:val_end]

    # -- Step 2: Train PD classifier ----------------------------------------
    logger.info("Training PD classifier")

    pd_config = PDModelConfig(
        hidden_sizes=(64, 32),
        dropout_rates=(0.2, 0.1),
        learning_rate=1e-3,
        epochs=50,
        batch_size=32,
        early_stopping_patience=10,
        random_seed=config.random_seed,
        cross_validation_folds=3,
    )
    pd_model = TFPDClassifier(config=pd_config)
    pd_model.fit(X_train, y_def_train)

    pd_eval = evaluate_pd_model(pd_model, X_test, y_def_test)
    logger.info("PD model — ROC-AUC: %.4f  Brier: %.4f", pd_eval.roc_auc, pd_eval.brier_score)

    # -- Step 3: Train spread regressor -------------------------------------
    logger.info("Training spread regressor")

    spread_config = SpreadModelConfig(
        hidden_sizes=(64, 32),
        dropout_rates=(0.2, 0.0),
        learning_rate=1e-3,
        epochs=80,
        batch_size=32,
        early_stopping_patience=15,
        random_seed=config.random_seed,
    )
    spread_model = TFSpreadRegressor(pd_model=pd_model, config=spread_config)
    spread_model.fit(X_train, y_spr_train)

    spread_eval = evaluate_spread_model(spread_model, X_test, y_spr_test)
    logger.info("Spread model — RMSE: %.1f bps  R²: %.4f", spread_eval.rmse_bps, spread_eval.r_squared)

    # -- Step 4: Train quantile regressors + conformal calibration ----------
    logger.info("Training quantile regressors")

    q_config = QuantileModelConfig(
        hidden_sizes=(64, 32),
        dropout_rates=(0.2, 0.0),
        learning_rate=1e-3,
        epochs=60,
        batch_size=32,
        early_stopping_patience=10,
        random_seed=config.random_seed,
    )
    lower_model = TFQuantileRegressor(quantile_alpha=0.025, config=q_config)
    upper_model = TFQuantileRegressor(quantile_alpha=0.975, config=q_config)
    lower_model.fit(X_train, y_spr_train)
    upper_model.fit(X_train, y_spr_train)

    conformal = ConformalPredictionInterval(lower_model, upper_model, coverage_target=0.95)
    conformal.calibrate(X_cal, y_spr_cal)
    coverage = conformal.empirical_coverage(X_test, y_spr_test)
    logger.info("Conformal 95%% interval — empirical coverage: %.2f  q_hat: %.2f", coverage, conformal.q_hat)

    # -- Step 5: OU calibration + Monte Carlo -------------------------------
    logger.info("Calibrating OU process")

    synthetic_spread_series = 200.0 + np.cumsum(rng.normal(0, 2, 300))
    ou_cal = OrnsteinUhlenbeckCalibrator(min_series_length=60, learning_rate=0.01, max_steps=2000)
    ou_params = ou_cal.fit(synthetic_spread_series)
    logger.info("OU params — kappa: %.4f  theta: %.2f  sigma: %.4f", ou_params.kappa, ou_params.theta, ou_params.sigma)

    vasicek_params = VasicekParameters(kappa=0.3, theta=3.5, sigma=0.5)
    mc_pricer = MonteCarloLoanPricer(
        ou_params=ou_params,
        vasicek_params=vasicek_params,
        n_simulations=config.monte_carlo_n_simulations,
        horizon_years=5.0,
        random_seed=config.random_seed,
    )
    mc_result = mc_pricer.simulate(current_spread_bps=200.0, current_treasury_yield_pct=4.0)
    logger.info(
        "Monte Carlo — mean: %.1f bps  95%% CI: [%.1f, %.1f]",
        mc_result.mean_spread, mc_result.ci_lower_95, mc_result.ci_upper_95,
    )

    # -- Step 6: Price a single loan ----------------------------------------
    logger.info("Pricing example loan")

    pricer = LoanPricer(
        pd_model=pd_model,
        spread_model=spread_model,
        quantile_models=(lower_model, upper_model),
    )

    example_loan = LoanPricingInput(
        financial_ratios={"debt_to_ebitda": 4.2, "interest_coverage_ratio": 3.5},
        loan_maturity_years=5.0,
        loan_size_mm=150.0,
        is_secured=True,
        industry_naics="5221",
        treasury_yield_pct=4.25,
    )
    result = pricer.price(example_loan)

    logger.info("--- Pricing Result ---")
    logger.info("  Estimated PD:       %.4f", result.estimated_pd)
    logger.info("  Credit Spread:      %.1f bps", result.credit_spread_bps)
    logger.info("  All-in Rate:        %.2f%%", result.all_in_rate_pct)
    logger.info("  Spread CI:          [%.1f, %.1f] bps", result.spread_ci_lower_bps, result.spread_ci_upper_bps)
    logger.info("  Internal Rating:    %s", result.internal_rating)

    # -- Step 7: Save models ------------------------------------------------
    config.models_dir.mkdir(parents=True, exist_ok=True)
    pd_model.save(config.models_dir / "pd_model.npz")
    spread_model.save(config.models_dir / "spread_model.npz")
    lower_model.save(config.models_dir / "quantile_lower.npz")
    upper_model.save(config.models_dir / "quantile_upper.npz")

    logger.info("All models saved to %s", config.models_dir)
    logger.info("Pipeline complete.")
