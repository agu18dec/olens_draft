"""Launcher-agnostic Inverted OLens job bodies.

The heavy compute lives here as plain functions taking an explicit ``root: Path``; the Modal
wrappers in ``scripts/ola/ola_modal.py`` and the Slurm CLI (``oracle_lens.pipeline.cluster_cli``)
both delegate to these — one body, two launchers, so the backends cannot silently diverge.
"""
