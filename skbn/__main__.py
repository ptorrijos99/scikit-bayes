"""``python -m skbn``: check the installation and compile the kernels."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys

from ._check import check_installation


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m skbn",
        description=(
            "Print the versions, fit and evaluate every model once (which"
            " compiles and caches its kernels) and verify the results."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="number of threads of the kernels (default: 1)",
    )
    args = parser.parse_args(argv)
    return 0 if check_installation(n_jobs=args.n_jobs) else 1


if __name__ == "__main__":
    sys.exit(main())
