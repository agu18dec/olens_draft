"""Shared constants for Inverted OLens jobs — single source for both launchers.

Duplicating these between the Modal file and a Slurm entrypoint is how the two silently diverge;
import from here instead. Values match the monorepo launchers as of the extraction.
"""

import os

MODEL_ID = "Qwen/Qwen3.6-27B"
DATASET_ID = "lmsys/lmsys-chat-1m"
LAYER = 44
T_MAX = 2048
N_MAX = 1024
MAX_PER_CONV = 8
MIN_RENDERED_TOKENS = 32
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "oracle-lens")
