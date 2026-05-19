"""Compatibility wrapper for the POSSM Stage-2 dropout ablation launcher."""

from possm_ssl.scripts.possm_stage2_dropout_ablation import *  # noqa: F401,F403
from possm_ssl.scripts.possm_stage2_dropout_ablation import main


if __name__ == "__main__":
    main()

