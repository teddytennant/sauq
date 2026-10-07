import jax
import jax.numpy as jnp
import pytest

from sauq import (
    confidence_trajectory,
    euclidean_gram,
    intrinsic_measures,
    layer_uncertainty,
    participation_ratio,
    sauq_score,
    select_head,
    signals_from_maps,
    tdo_symmetric,
)


def row_stochastic(key, n):
    raw = jax.random.uniform(key, (n, n), dtype=jnp.float64) + 0.05
    return raw / raw.sum(axis=1, keepdims=True)


def test_uniform_measures_and_pr():
    n = 4
    attention = jnp.ones((n, n)) / n
    mu, pi = intrinsic_measures(attention)
    assert jnp.allclose(mu, pi)
    assert jnp.allclose(pi, jnp.ones(n) / n)
    sym = tdo_symmetric(attention)
    assert jnp.allclose(sym @ jnp.ones(n), jnp.ones(n), atol=1e-8)
    # one eigenvalue 1, the rest 0, so PR = 1/n
    assert float(participation_ratio(attention)) == pytest.approx(1.0 / n, abs=1e-8)


def test_random_row_stochastic_has_leading_eigenvalue_one():
    attention = row_stochastic(jax.random.PRNGKey(0), 5)
    mu, _pi = intrinsic_measures(attention)
    assert jnp.all(mu > 0)
    # S = M^{1/2} D M^{-1/2}, so the constant eigenvector of D is sqrt(mu) for S.
    sym = tdo_symmetric(attention)
    direction = jnp.sqrt(mu)
    assert jnp.allclose(sym @ direction, direction, atol=1e-5)
    leading = jnp.linalg.eigvalsh(sym)[-1]
    assert float(leading) == pytest.approx(1.0, abs=1e-4)


def test_non_expansion():
    attention = row_stochastic(jax.random.PRNGKey(1), 6)
    mu, pi = intrinsic_measures(attention)
    key = jax.random.PRNGKey(2)
    for i in range(5):
        f = jax.random.normal(jax.random.fold_in(key, i), (6,), dtype=jnp.float64)
        left = jnp.sum(pi * (attention @ f) ** 2)
        right = jnp.sum(mu * f ** 2)
        assert float(left) <= float(right) + 1e-6


def test_causal_sink_euclidean_spectrum():
    n = 8
    attention = jnp.zeros((n, n)).at[:, 0].set(1.0)
    gram = euclidean_gram(attention)
    expect = jnp.zeros((n, n)).at[0, 0].set(float(n))
    assert jnp.allclose(gram, expect)
    assert float(jnp.linalg.eigvalsh(gram)[-1]) == pytest.approx(float(n))


def test_trace_pr_matches_eigenvalues():
    attention = row_stochastic(jax.random.PRNGKey(3), 5)
    sym = tdo_symmetric(attention)
    evals = jnp.linalg.eigvalsh(sym)
    n = evals.shape[0]
    from_eigs = (jnp.sum(evals) ** 2) / (n * jnp.sum(evals ** 2))
    assert float(participation_ratio(attention)) == pytest.approx(float(from_eigs), abs=1e-5)


def test_centered_pr_drops_the_leading_one():
    attention = row_stochastic(jax.random.PRNGKey(4), 5)
    sym = tdo_symmetric(attention)
    evals = jnp.sort(jnp.linalg.eigvalsh(sym))
    rest = evals[:-1]
    expect = (jnp.sum(rest) ** 2) / ((rest.shape[0]) * jnp.sum(rest ** 2))
    got = participation_ratio(attention, centered=True)
    assert float(got) == pytest.approx(float(expect), abs=1e-5)
    assert float(got) > 0.0


def test_confidence_recurrence():
    probs = jnp.array([0.5, 0.4, 0.8, 0.2])
    signals = jnp.array([0.9, 0.5, 0.25, 1.0])
    alpha = 0.2
    c = confidence_trajectory(probs, signals, alpha=alpha)
    hand = [0.5]
    for t in range(1, 4):
        hand.append(alpha * float(probs[t]) + (1 - alpha) * float(signals[t]) * hand[-1])
    assert jnp.allclose(c, jnp.array(hand), atol=1e-6)


def test_sequence_score_is_the_max_layer():
    # Layer 0 stays confident. Layer 1 collapses, so its uncertainty is larger.
    probs = jnp.array([[0.8, 0.8, 0.8], [0.8, 0.8, 0.8]])
    signals = jnp.array([[1.0, 1.0, 1.0], [1.0, 0.05, 0.05]])
    u0 = layer_uncertainty(probs[0], signals[0])
    u1 = layer_uncertainty(probs[1], signals[1])
    assert float(u1) > float(u0)
    assert float(sauq_score(probs, signals)) == pytest.approx(float(u1))


def test_head_selection_prefers_the_spread_block():
    t = 4
    rank1 = jnp.ones((t, 1)) @ jnp.array([[1.0, 0.0, 0.0, 0.0]])
    spread = jnp.eye(t)
    mixed = 0.5 * rank1 + 0.5 * spread
    blocks = jnp.stack([rank1, spread, mixed])
    index, scores = select_head(blocks)
    assert index == int(jnp.argmax(scores))
    assert float(scores[1]) > float(scores[0])


def test_signals_are_intrinsic_pr_and_roll_forward():
    n = 3
    maps = []
    for scale in (0.2, 0.5, 0.8):
        raw = jnp.full((n, n), scale) + jnp.eye(n)
        maps.append(raw / raw.sum(axis=1, keepdims=True))
    maps = jnp.stack(maps)
    signals = signals_from_maps(maps)
    for i in range(3):
        assert float(signals[i]) == pytest.approx(float(participation_ratio(maps[i])))
    probs = jnp.array([0.6, 0.7, 0.4])
    got = layer_uncertainty(probs, signals)
    c = confidence_trajectory(probs, signals)
    expect = 1.0 - jnp.mean(jnp.log(c))
    assert float(got) == pytest.approx(float(expect))
