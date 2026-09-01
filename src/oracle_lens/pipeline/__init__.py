"""The multilayer oracle-lens pipeline: data generation, AR/AO SFT, distillation, RL reward.

Data flows: seed prep → on-policy rollouts (SGLang) → multilayer activation capture
(``multilayer``) → whiteners → AR training (``multilayer_reconstructor``, ``ar_loader``) →
AO pools/arout (``ao_pool``, ``ao_arout``, ``recon_precompute``) → AO SFT (``soft_token_sft``,
``ao_ladder``, ``ao_train``, ``gt_train``) → distillation (``distill``, ``distill_shards``) →
GRPO RL (``rl_reward`` + ``scripts/rl/``).

Launch scripts live in ``scripts/{datagen,ar,ao,distill,rl}``; stage-by-stage commands in
``docs/pipeline.md``.
"""
