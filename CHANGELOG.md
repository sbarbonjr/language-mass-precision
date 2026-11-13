# Changelog

## 0.1.0 — unreleased

First release of the library, extracted from the ICPM 2026 evaluation pipeline.

- Exact LMP over a compiled DFA: finite sum, sparse solve (geometric prior), certified intervals.
- Live-mass tail bound for certified intervals: `D_K <= D <= D_K + s_{K+1} * sum_{n>K} rho_n`.
  It reproduces the paper pipeline's geometric values exactly and gives much tighter power-law intervals.
- Generic activity prior (uniform, Laplace-smoothed empirical, or user mapping) for every length prior.
- Alphabet built from model-*language* labels, so dead branches never change the prior.
- Bounded-playout baseline (`method="playout"`).
- Trace-level imprecision attribution by best-first DFA search, valid on infinite languages.
- PM4Py-style `algorithm.apply` with `Variants`/`Parameters`; scikit-learn style `LanguageMassPrecision`.
- Toy datasets from the paper's controlled evaluation; seven executed tutorials.
