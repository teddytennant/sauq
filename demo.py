"""Synthetic attention maps: uniform, causal sink, copying head."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import jax.numpy as jnp

from sauq import euclidean_gram, participation_ratio
from sauq.estimator import confidence_trajectory, layer_uncertainty

n = 6
uniform = jnp.ones((n, n)) / n
sink = jnp.zeros((n, n)).at[:, 0].set(1.0)
copy = jnp.eye(n)
print("uniform intrinsic PR", float(participation_ratio(uniform)))
print("sink euclidean leading", float(jnp.linalg.eigvalsh(euclidean_gram(sink))[-1]))
print("copy intrinsic PR", float(participation_ratio(copy)))
probs = jnp.array([0.7, 0.6, 0.5, 0.4])
signals = jnp.array([0.4, 0.3, 0.2, 0.15])
print("confidence", confidence_trajectory(probs, signals))
print("U", float(layer_uncertainty(probs, signals)))
