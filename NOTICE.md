# Third-party code

The MIT licence in `LICENSE` covers this repository's own code. These vendored trees are not
covered by it and keep their own terms:

- **`src/jlens/`** — vendored from [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens),
  Apache License 2.0. Its licence is in `src/jlens/LICENSE`, its provenance in
  `src/jlens/VENDORED.md`.
- **`vendor/mytorch-lightning/`** — vendored from jordan-benjamin/mytorch-lightning, a **private**
  repository, carried here so the repo builds without access to it. No licence is granted here and
  all rights are reserved by its owner; see `vendor/mytorch-lightning/VENDORED.md`, which asks for
  the owner's permission before this repo is published.
