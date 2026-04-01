"""Unit tests for decomposed PolicyTrainer sub-methods.

Tests use concrete float64 tensors and assert specific numerical outputs.
"""

import math

import pytest
import tensorflow as tf

from financial_forecast.training.policy_trainer import PolicyTrainer, PolicyLossScales


def _f64(values):
    return tf.constant(values, dtype=tf.float64)


class TestComputeLossScales:
    """Tests for PolicyTrainer._compute_loss_scales()."""

    def test_std_mode_computes_correct_scales(self):
        """Verify each scale = std(series) + eps with known data.

        Using [1.0, 3.0] → std = 1.0 for a simple 2-element series.
        Using [10.0, 20.0, 30.0] → std = std([10,20,30]) for a 3-element.
        """
        # 2-element series: std([1, 3]) = 1.0
        two_elem = _f64([1.0, 3.0])
        # 3-element series: std([10, 20, 30]) = sqrt(200/3) ≈ 8.16497
        three_elem = _f64([10.0, 20.0, 30.0])
        # constant series: std([5, 5, 5]) = 0.0 → scale = eps
        constant = _f64([5.0, 5.0, 5.0])

        eps = 1e-12
        expected_two_std = float(tf.math.reduce_std(two_elem).numpy())
        expected_three_std = float(tf.math.reduce_std(three_elem).numpy())

        scales = PolicyTrainer._compute_loss_scales(
            "std",
            delta_nca_true=two_elem,
            depr_true=three_elem,
            adv_ps_true=two_elem,
            adv_pp_true=two_elem,
            ar=three_elem,
            ap=three_elem,
            inv=two_elem,
            cash=_f64([100.0, 200.0, 300.0]),
            ims=_f64([10.0, 20.0, 30.0]),
            tax=constant,
            div_true=two_elem,
            bb=three_elem,
            logit_cr_hist=two_elem,
            eff_st_debt=three_elem,
            opex=two_elem,
        )

        assert isinstance(scales, PolicyLossScales)

        # growth uses delta_nca_true = [1.0, 3.0] → std = 1.0
        assert float(scales.growth) == pytest.approx(expected_two_std + eps)

        # depr uses three_elem = [10, 20, 30]
        assert float(scales.depr) == pytest.approx(expected_three_std + eps)

        # tax uses constant = [5, 5, 5] → std = 0.0, scale = eps
        assert float(scales.tax) == pytest.approx(eps)

        # tl = std(cash + ims) = std([110, 220, 330])
        expected_tl_std = float(tf.math.reduce_std(_f64([110.0, 220.0, 330.0])).numpy())
        assert float(scales.tl) == pytest.approx(expected_tl_std + eps)

    def test_none_mode_returns_all_ones(self):
        """All scales should be 1.0 when mode='none'."""
        dummy = _f64([1.0, 2.0, 3.0])

        scales = PolicyTrainer._compute_loss_scales(
            "none",
            delta_nca_true=dummy,
            depr_true=dummy,
            adv_ps_true=dummy,
            adv_pp_true=dummy,
            ar=dummy,
            ap=dummy,
            inv=dummy,
            cash=dummy,
            ims=dummy,
            tax=dummy,
            div_true=dummy,
            bb=dummy,
            logit_cr_hist=dummy,
            eff_st_debt=dummy,
            opex=dummy,
        )

        assert float(scales.growth) == pytest.approx(1.0)
        assert float(scales.depr) == pytest.approx(1.0)
        assert float(scales.adv_ps) == pytest.approx(1.0)
        assert float(scales.adv_pp) == pytest.approx(1.0)
        assert float(scales.ar) == pytest.approx(1.0)
        assert float(scales.ap) == pytest.approx(1.0)
        assert float(scales.inv) == pytest.approx(1.0)
        assert float(scales.tl) == pytest.approx(1.0)
        assert float(scales.cash) == pytest.approx(1.0)
        assert float(scales.tax) == pytest.approx(1.0)
        assert float(scales.div) == pytest.approx(1.0)
        assert float(scales.bb) == pytest.approx(1.0)
        assert float(scales.cost_ratio) == pytest.approx(1.0)
        assert float(scales.eff_st) == pytest.approx(1.0)
        assert float(scales.opex) == pytest.approx(1.0)

    def test_invalid_mode_raises_value_error(self):
        """Unknown loss_scale_mode should raise ValueError."""
        dummy = _f64([1.0, 2.0])

        with pytest.raises(ValueError, match="Unsupported loss_scale_mode='bad'"):
            PolicyTrainer._compute_loss_scales(
                "bad",
                delta_nca_true=dummy,
                depr_true=dummy,
                adv_ps_true=dummy,
                adv_pp_true=dummy,
                ar=dummy,
                ap=dummy,
                inv=dummy,
                cash=dummy,
                ims=dummy,
                tax=dummy,
                div_true=dummy,
                bb=dummy,
                logit_cr_hist=dummy,
                eff_st_debt=dummy,
                opex=dummy,
            )

    def test_single_element_series_returns_eps(self):
        """A single-element series has std=0 → scale should be eps."""
        single = _f64([42.0])
        eps = 1e-12

        scales = PolicyTrainer._compute_loss_scales(
            "std",
            delta_nca_true=single,
            depr_true=single,
            adv_ps_true=single,
            adv_pp_true=single,
            ar=single,
            ap=single,
            inv=single,
            cash=single,
            ims=single,
            tax=single,
            div_true=single,
            bb=single,
            logit_cr_hist=single,
            eff_st_debt=single,
            opex=single,
        )

        assert float(scales.growth) == pytest.approx(eps)
        assert float(scales.opex) == pytest.approx(eps)
