"""End-to-end credit rating experiment — load, train, evaluate, case studies."""

from credit_rating.experiment import (
    CreditRatingExperimentConfig,
    run_credit_rating_experiment,
)

if __name__ == "__main__":
    run_credit_rating_experiment(CreditRatingExperimentConfig())
