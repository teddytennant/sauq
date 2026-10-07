# sauq

JAX implementation of the Token Difference Operator and the SAUQ confidence recurrence ([arXiv:2610.06012](https://arxiv.org/abs/2610.06012)).

## Install

```bash
pip install -e ".[test]"
```

## Run

```bash
JAX_PLATFORMS=cpu python -m pytest -q
JAX_PLATFORMS=cpu python demo.py
```

## What does not match the paper

No Llama or Qwen, and no LongBench-style evaluation. Attention maps are synthetic. `alpha = 0.2` and the recurrence `c_1 = p_1`, `c_t = alpha p_t + (1 - alpha) s_t c_{t-1}` match Section 5.1. The sequence score is the max layer-wise uncertainty. The paper's LLaMA experiment used the middle third of layers; this package does not hard-code that slice.

If some key has `mu_j = 0`, the intrinsic operator is restricted to the support of `mu` instead of adding an epsilon floor. The causal-sink claim that is tested is the Euclidean one: `A^T A = n e_0 e_0^T`. Head selection uses `PR(B B^T)` on the generated block, as written in Section 5.1, and the spectral signal `s_t` is the intrinsic participation ratio.
