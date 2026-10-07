"""Token Difference Operator and participation ratio (arXiv:2610.06012).

If some mu_j is zero the intrinsic operator is singular. The symmetric matrix
is restricted to the support {j : mu_j > 0}. Participation ratio then uses the
support size as n. That is a restriction, not an epsilon floor.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def intrinsic_measures(attention):
    """pi = 1/n, mu = A^T pi. A is row-stochastic, rows are queries."""
    attention = jnp.asarray(attention, dtype=jnp.float64)
    n = attention.shape[0]
    pi = jnp.ones(n, dtype=jnp.float64) / n
    mu = attention.T @ pi
    return mu, pi


def support_mask(mu, tol=0.0):
    return mu > tol


def tdo_symmetric(attention):
    """S = M^{-1/2} A^T Pi A M^{-1/2} on the support of mu.

    S is symmetric and similar to the Token Difference Operator, so it has the
    same eigenvalues.
    """
    attention = jnp.asarray(attention, dtype=jnp.float64)
    mu, pi = intrinsic_measures(attention)
    mask = support_mask(mu)
    mu_s = mu[mask]
    keys = attention[:, mask]
    gram = (keys * jnp.sqrt(pi)[:, None]).T @ (keys * jnp.sqrt(pi)[:, None])
    inv_sqrt = 1.0 / jnp.sqrt(mu_s)
    return inv_sqrt[:, None] * gram * inv_sqrt[None, :]


def participation_ratio(attention, centered=False):
    """PR = (sum lambda)^2 / (n sum lambda^2) = tr(S)^2 / (n ||S||_F^2).

    Centered PR drops the leading eigenvalue. For a row-stochastic map with
    full support that eigenvalue is 1, and the denominator uses n-1.
    """
    sym = tdo_symmetric(attention)
    n = sym.shape[0]
    tr = jnp.trace(sym)
    fro2 = jnp.sum(sym ** 2)
    if not centered:
        return (tr ** 2) / (n * fro2)
    evals = jnp.linalg.eigvalsh(sym)
    rest = evals[:-1]
    denom_n = jnp.maximum(n - 1, 1)
    return (jnp.sum(rest) ** 2) / (denom_n * jnp.sum(rest ** 2))


def euclidean_gram(attention):
    attention = jnp.asarray(attention, dtype=jnp.float64)
    return attention.T @ attention


def gram_participation_ratio(block):
    """Euclidean PR of B B^T, the head-selection proxy in Section 5.1."""
    block = jnp.asarray(block, dtype=jnp.float64)
    gram = block @ block.T
    n = gram.shape[0]
    tr = jnp.trace(gram)
    fro2 = jnp.sum(gram ** 2)
    return (tr ** 2) / (n * fro2)


def select_head(blocks):
    """blocks (H, T, T). Returns the head with the largest PR(B B^T)."""
    scores = jax_vmap_pr(blocks)
    return int(jnp.argmax(scores)), scores


def jax_vmap_pr(blocks):
    import jax

    return jax.vmap(gram_participation_ratio)(blocks)
