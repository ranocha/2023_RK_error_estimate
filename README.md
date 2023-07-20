# Stability of step size control based on a posteriori error estimates

[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/TODO.svg)](https://doi.org/TODO)

This repository contains information and code to reproduce the results presented in the
article
```bibtex
@online{ranocha2023stability,
  title={Stability of step size control based on a posteriori error estimates},
  author={Ranocha, Hendrik and Giesselmann, Jan},
  year={2023},
  month={07},
  doi={TODO},
  eprint={TODO},
  eprinttype={arxiv},
  eprintclass={math.NA}
}
```

If you find these results useful, please cite the article mentioned above. If you
use the implementations provided here, please **also** cite this repository as
```bibtex
@misc{ranocha2023stepRepro,
  title={Reproducibility repository for
         "{S}tability of step size control based on a posteriori error estimates"},
  author={Ranocha, Hendrik and Giesselmann, Jan},
  year={2023},
  howpublished={\url{https://github.com/ranocha/2023_RK_error_estimate}},
  doi={TODO}
}
```

## Abstract

A posteriori error estimates based on residuals can be used for reliable error
control of numerical methods. Here, we consider them in the context of ordinary
differential equations and Runge-Kutta methods. In particular, we take the
approach of Dedner & Giesselmann (2016) and investigate it when used to select
the time step size. We focus on step size control stability when combined with
explicit Runge-Kutta methods and demonstrate that a standard I controller is
unstable while more advanced PI and PID controllers can be designed to be
stable. We compare the stability properties of residual-based
estimators and classical error estimators based on an embedded Runge-Kutta method
both analytically and in numerical experiments.


## Numerical experiments

To reproduce the numerical experiments presented in this article, you need
to install [Julia](https://julialang.org/). The numerical experiments presented
in this article were performed using Julia v1.9.2.

First, you need to download this repository, e.g., by cloning it with `git`
or by downloading an archive via the GitHub interface. Then, you need to start
Julia in the `code` directory and follow the instructions described in the
`README.md` file there.


## Authors

- [Hendrik Ranocha](https://ranocha.de) (University of Hamburg, Germany)
- Jan Giesselmann (TU Darmstadt, Germany)


## Disclaimer

Everything is provided as is and without warranty. Use at your own risk!
