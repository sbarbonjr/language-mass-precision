# Contributing

Thanks for helping improve `language-mass-precision`.

## Setup

```bash
pip install -e ".[dev,tutorials]"
pytest
ruff check src tests
```

## Conventions

- **Layout follows PM4Py.** Algorithms live in `algo/<task>/algorithm.py` with `Variants` and `Parameters` enums and an `apply(...)` entry point. Variants go in `algo/<task>/variants/`. Data structures go in `objects/`.
- **No silent approximations.** A computation that cannot certify its answer must say so through `Status` (`interval`, `bounded`, `incomplete`), never by returning a plain number.
- **Language, not structure.** Anything that changes a score must be a property of the model language. Add a language-equivalence test when touching compilation or the alphabet.
- **Tests first for the engine.** Changes to `algo/denominator.py` or `algo/conversion/` need a test against an explicitly enumerated language or a known closed form.
- Keep the public API in `__init__.py` small and documented in the README.

## Tutorials

Notebooks in `tutorials/` are committed with outputs. Re-execute any notebook you edit (`jupyter nbconvert --to notebook --execute --inplace <file>`) so its outputs match the code.
