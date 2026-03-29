"""Save and load model parameters to/from .npz files."""

import os

import numpy as np
import tensorflow as tf


def _save_module_params(params: dict, module: tf.Module, prefix: str = "") -> None:
    """Save all trainable variables from a tf.Module."""
    for var in module.trainable_variables:
        key = prefix + var.name.split(":")[0].replace("/", "_")
        params[key] = float(var.numpy())
    # Also save non-trainable variables that are needed
    for var in module.variables:
        if not var.trainable:
            key = prefix + var.name.split(":")[0].replace("/", "_")
            params[key] = float(var.numpy())


def save_parameters(model: tf.Module, path: str) -> None:
    """Serialize every learnable parameter to a NumPy ``.npz`` archive."""
    params = {}

    # Save all submodule params via tf.Module variable tracking
    for var in model.trainable_variables:
        key = var.name.split(":")[0]
        params[key] = float(var.numpy())

    # Non-trainable variables (like sales_offset)
    for var in model.variables:
        if not var.trainable:
            key = var.name.split(":")[0]
            params[key] = float(var.numpy())

    # Metadata
    params["base_year"] = model.base_year
    params["amount_scale"] = model.amount_scale

    np.savez(path, **params)


def load_parameters(model: tf.Module, path: str) -> None:
    """Restore model parameters from a previously saved ``.npz`` archive."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parameter file not found: {path}")
    data = np.load(path, allow_pickle=True)

    # Load all variables by name matching
    all_vars = {v.name.split(":")[0]: v for v in model.variables}
    for key in data.files:
        if key in ("base_year", "amount_scale"):
            continue
        if key in all_vars:
            all_vars[key].assign(float(data[key]))

    # Metadata
    if "base_year" in data:
        model.base_year = int(data["base_year"])
    if "amount_scale" in data:
        model.amount_scale = float(data["amount_scale"])
