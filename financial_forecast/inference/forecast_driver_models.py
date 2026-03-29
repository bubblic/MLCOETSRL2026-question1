"""Forecast driver models for sales and inflation.

Pluggable into :class:`ForecastPipeline` to generate the exogenous
inputs (sales trajectory, inflation trajectory) that drive the
balance-sheet simulation.

Each model is fully initialized at construction time — ``forecast``
is available immediately after ``__init__``.
"""

from abc import ABC, abstractmethod

import tensorflow as tf


class SalesForecastModel(ABC):
    """Abstract base for sales forecast models."""

    forecast: tf.Tensor

    @property
    @abstractmethod
    def n_years(self) -> int:
        """Number of forecast years."""


class InflationForecastModel(ABC):
    """Abstract base for inflation forecast models."""

    forecast: tf.Tensor

    @property
    @abstractmethod
    def n_years(self) -> int:
        """Number of forecast years."""


class LinearSalesForecast(SalesForecastModel):
    """Linear extrapolation of the average year-over-year sales delta.

    Args:
        historical_sales_usd: 1-D tensor of historical sales in USD.
        forecast_years: Number of years to forecast.
    """

    def __init__(
        self,
        historical_sales_usd: tf.Tensor,
        forecast_years: int,
    ):
        sales = historical_sales_usd
        avg_growth = tf.reduce_mean(sales[1:] - sales[:-1])
        self.forecast = tf.constant(
            [float(sales[-1] + avg_growth * i) for i in range(forecast_years)],
            dtype=tf.float64,
        )
        self._n_years = forecast_years

    @property
    def n_years(self) -> int:
        return self._n_years


class ConstantInflationForecast(InflationForecastModel):
    """Last observed rate for year 1, then constant rate thereafter.

    If historical inflation is all zeros, forecasts zero inflation.

    Args:
        historical_inflation: 1-D tensor of annual inflation rates.
        forecast_years: Number of years to forecast.
        default_rate: Constant rate for years 2+. Defaults to 3%.
    """

    def __init__(
        self,
        historical_inflation: tf.Tensor,
        forecast_years: int,
        default_rate: float = 0.03,
    ):
        inf = historical_inflation
        if tf.reduce_all(inf == 0.0):
            self.forecast = tf.zeros([forecast_years], dtype=tf.float64)
        else:
            self.forecast = tf.concat(
                [
                    inf[-1:],
                    tf.fill(
                        [forecast_years - 1],
                        tf.constant(default_rate, dtype=tf.float64),
                    ),
                ],
                axis=0,
            )
        self._n_years = forecast_years

    @property
    def n_years(self) -> int:
        return self._n_years
