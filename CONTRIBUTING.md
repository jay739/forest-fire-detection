# Contributing to forest-fire-detection

Thanks for considering a contribution. Issues, discussions and PRs are all
welcome.

## Getting set up

Python 3.10 or newer. `pip install -r requirements.txt` (CPU wheels; see the README for CUDA). Download the FLAME dataset and point `--path_rgb` and `--path_ir` at its 254p folders.

## Running the checks

`pip install -r requirements.txt pytest` then `pytest -q tests`. The tests need no dataset, GPU or network. CI runs them on every PR.

## Ground rules

- Open an issue or discussion before large changes, so effort isn't wasted.
- Keep PRs focused: one change per PR.
- New behavior needs a test where the repo has a test suite.
- CI must pass before merge.

## Licensing

By contributing you agree your contributions are licensed under the repo's
MIT license.
