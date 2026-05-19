"""Compatibility wrapper for the POSSM Stage-2 hyperparameter sweep launcher."""

from possm_ssl.scripts.possm_stage2_hyperparam_sweep import *  # noqa: F401,F403
from possm_ssl.scripts.possm_stage2_hyperparam_sweep import main


if __name__ == "__main__":
    main()

