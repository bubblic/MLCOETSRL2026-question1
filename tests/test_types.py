"""Tests for types.py TypedDict contracts and boundary validators."""

import pytest
import tensorflow as tf

from financial_forecast.types import (
    validate_recurrent_state,
    validate_forecast_inputs,
    _RECURRENT_STATE_KEYS,
    _FORECAST_INPUTS_KEYS,
)


def _make_valid_recurrent_state():
    """Build a valid RecurrentState dict with all 14 keys."""
    return {k: tf.constant(0.0, dtype=tf.float64) for k in _RECURRENT_STATE_KEYS}


def _make_valid_forecast_inputs():
    """Build a valid ForecastInputs dict with all 3 keys."""
    return {k: tf.constant(0.0, dtype=tf.float64) for k in _FORECAST_INPUTS_KEYS}


class TestValidateRecurrentState:

    def test_valid_state_passes_silently(self):
        """A dict with exactly the 14 expected keys should not raise."""
        state = _make_valid_recurrent_state()
        validate_recurrent_state(state)

    def test_missing_key_raises_key_error(self):
        """Removing one key should raise KeyError naming the missing key."""
        state = _make_valid_recurrent_state()
        del state["dividends"]

        with pytest.raises(KeyError, match="missing keys.*dividends"):
            validate_recurrent_state(state)

    def test_extra_key_raises_key_error(self):
        """Adding an unknown key should raise KeyError naming the extra key."""
        state = _make_valid_recurrent_state()
        state["net_incoem"] = tf.constant(0.0)  # typo

        with pytest.raises(KeyError, match="unexpected keys.*net_incoem"):
            validate_recurrent_state(state)

    def test_misspelled_key_reports_both_missing_and_extra(self):
        """A misspelled key produces both a missing and an extra key error.

        The validator checks missing first, so the error message reports
        the missing key.
        """
        state = _make_valid_recurrent_state()
        del state["equity"]
        state["equitty"] = tf.constant(0.0)  # typo

        with pytest.raises(KeyError, match="missing keys.*equity"):
            validate_recurrent_state(state)

    def test_context_label_appears_in_error(self):
        """The context string should appear in the error message."""
        state = _make_valid_recurrent_state()
        del state["nca"]

        with pytest.raises(KeyError, match=r"\[prepare\].*missing"):
            validate_recurrent_state(state, context="prepare")

    def test_empty_dict_raises_key_error(self):
        """An empty dict should raise, listing all 14 missing keys."""
        with pytest.raises(KeyError, match="missing keys"):
            validate_recurrent_state({})

    def test_expected_key_count_is_14(self):
        """Sanity check: RecurrentState should have exactly 14 keys."""
        assert len(_RECURRENT_STATE_KEYS) == 14


class TestValidateForecastInputs:

    def test_valid_inputs_passes_silently(self):
        """A dict with exactly the 3 expected keys should not raise."""
        inputs = _make_valid_forecast_inputs()
        validate_forecast_inputs(inputs)

    def test_missing_key_raises_key_error(self):
        """Removing 'year' should raise KeyError."""
        inputs = _make_valid_forecast_inputs()
        del inputs["year"]

        with pytest.raises(KeyError, match="missing keys.*year"):
            validate_forecast_inputs(inputs)

    def test_extra_key_raises_key_error(self):
        """Adding 'purchases_t' (wrong contract) should raise KeyError."""
        inputs = _make_valid_forecast_inputs()
        inputs["purchases_t"] = tf.constant(0.0)

        with pytest.raises(KeyError, match="unexpected keys.*purchases_t"):
            validate_forecast_inputs(inputs)

    def test_context_label_appears_in_error(self):
        """The context string should appear in the error message."""
        with pytest.raises(KeyError, match=r"\[forecast_step\]"):
            validate_forecast_inputs({}, context="forecast_step")

    def test_expected_key_count_is_3(self):
        """Sanity check: ForecastInputs should have exactly 3 keys."""
        assert len(_FORECAST_INPUTS_KEYS) == 3
