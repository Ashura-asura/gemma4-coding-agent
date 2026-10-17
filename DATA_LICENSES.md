# Data and third-party licences

This repository redistributes derived data. Check these before publishing or
entering a competition that requires a specific licence.

* `data/*_tasks.jsonl`, `data/sft_trajectories*.jsonl` are derived from the
  **SWE-bench** test split (issue text, test patches) and contain excerpts of
  source files from the upstream projects (via `read_file` observations and
  gold diffs). Each upstream project keeps its own licence: Django, sympy,
  Sphinx, scikit-learn, matplotlib (BSD/PSF-style), pytest (MIT), astropy
  (BSD-3), xarray (Apache-2.0), and **pylint (GPL-2.0)** — the pylint excerpts
  deserve legal review before redistribution. Verify each repo's current licence.
* `LICENSE` (Apache-2.0) was added by the audit as a **placeholder for the
  code**; confirm it matches the licence your competition/organisation
  requires before release.
* Model weights (Gemma 4) are governed by Google's Gemma terms, not this repo's licence.
* `data/sft_trajectories.jsonl` is ~59 MB; consider Git LFS or a Kaggle dataset
  instead of committing it, and ship only the compacted file.
