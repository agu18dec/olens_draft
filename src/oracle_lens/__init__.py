"""Oracle lens: train a model to read another model's activations in natural language.

Two models, both LoRA adapters on a frozen Qwen3.6-27B base:

- **AR (activation reconstructor)** — text span → the residual activation that preceded it,
  at many layers in one forward (``oracle_lens.pipeline.multilayer_reconstructor``). Trained
  first, on on-policy rollouts, then frozen forever; it is the reward model for everything after.
- **AO (activation oracle)** — activation → text. One activation vector replaces the embedding
  of a marker slot; the model is SFT-trained to verbalize it
  (``oracle_lens.pipeline.soft_token_sft``), then distilled to a multi-bullet student and
  RL-tuned with GRPO against the frozen AR (``scripts/rl/``).

Pipeline order and commands: ``docs/pipeline.md``. Hard-won operational rules:
``docs/failure_modes.md``.
"""
