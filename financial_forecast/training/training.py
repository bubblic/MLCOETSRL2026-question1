"""Training entry points for the trainable financial model.

This module exposes thin wrapper functions that delegate to the
corresponding methods on a ``TrainableFinancialModel`` instance,
keeping the public training API decoupled from the model class itself.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from financial_forecast.models.bayesian_model import BayesianFinancialModel as TrainableFinancialModel


def train_simple_policies(
    model: TrainableFinancialModel, *args: Any, **kwargs: Any
) -> None:
    """Train simple policy parameters on a model instance.

    Args:
        model: A ``TrainableFinancialModel`` whose simple-policy parameters
            will be optimised in-place.
        *args: Positional arguments forwarded to
            ``model.train_simple_policies``.
        **kwargs: Keyword arguments forwarded to
            ``model.train_simple_policies``.
    """
    model.train_simple_policies(*args, **kwargs)


def train_structural_parameters(
    model: TrainableFinancialModel, *args: Any, **kwargs: Any
) -> None:
    """Train structural parameters on a model instance.

    Args:
        model: A ``TrainableFinancialModel`` whose structural parameters
            will be optimised in-place.
        *args: Positional arguments forwarded to
            ``model.train_structural_parameters``.
        **kwargs: Keyword arguments forwarded to
            ``model.train_structural_parameters``.
    """
    model.train_structural_parameters(*args, **kwargs)


__all__ = ["train_simple_policies", "train_structural_parameters"]
