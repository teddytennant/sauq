"""SAUQ confidence recurrence (arXiv:2610.06012, Section 5.1)."""

from __future__ import annotations

import jax.numpy as jnp

from sauq.tdo import participation_ratio, select_head


def confidence_trajectory(probs, signals, alpha=0.2):
    """c_1 = p_1, c_t = alpha p_t + (1 - alpha) s_t c_{t-1} for t > 1."""
    probs = jnp.asarray(probs, dtype=jnp.float64)
    signals = jnp.asarray(signals, dtype=jnp.float64)
    c = [probs[0]]
    for t in range(1, probs.shape[0]):
        c.append(alpha * probs[t] + (1.0 - alpha) * signals[t] * c[-1])
    return jnp.stack(c)


def layer_uncertainty(probs, signals, alpha=0.2):
    """U = 1 - mean(log c_t)."""
    c = confidence_trajectory(probs, signals, alpha=alpha)
    return 1.0 - jnp.mean(jnp.log(c))


def sauq_score(layer_probs, layer_signals, alpha=0.2):
    """Sequence score is the max layer-wise uncertainty."""
    layer_probs = jnp.asarray(layer_probs, dtype=jnp.float64)
    layer_signals = jnp.asarray(layer_signals, dtype=jnp.float64)
    scores = [
        layer_uncertainty(layer_probs[i], layer_signals[i], alpha=alpha)
        for i in range(layer_probs.shape[0])
    ]
    return jnp.max(jnp.stack(scores))


def signals_from_maps(maps):
    """Intrinsic PR of each attention map. maps is (T, n, n) or a list."""
    return jnp.stack([participation_ratio(m) for m in maps])


def select_and_score(blocks, probs, maps, alpha=0.2):
    """Pick the head by Euclidean PR of B B^T, then score the intrinsic signals."""
    index, _scores = select_head(blocks)
    signals = signals_from_maps(maps)
    return index, layer_uncertainty(probs, signals, alpha=alpha)
