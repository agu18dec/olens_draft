"""NNOMP: planted-combo recovery, non-negativity, masking, stopping, dictionary sampling."""

import random

import pytest
import torch

from oracle_lens.core.dictionary import (
    fits_fn,
    merge_deduped,
    phrases_at_positions,
    sample_dictionary_phrases,
    sample_dictionary_positions,
)
from oracle_lens.core.nnomp import half_mask, nnomp_batch


def _unit_dictionary(n: int = 512, d: int = 64, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    dictionary = torch.randn(n, d, generator=gen)
    out: torch.Tensor = dictionary / dictionary.norm(dim=-1, keepdim=True)
    return out


def test_recovers_planted_nonnegative_combo() -> None:
    dictionary = _unit_dictionary()
    true_atoms = torch.tensor([[7, 100, 300], [5, 50, 499]])
    true_coeffs = torch.tensor([[2.0, 1.0, 0.5], [1.5, 0.7, 3.0]])
    x = torch.einsum("bk,bkd->bd", true_coeffs, dictionary[true_atoms])
    result = nnomp_batch(x, dictionary, max_atoms=8)
    for row in range(2):
        got = {int(a) for a in result.atoms[row] if a >= 0}
        assert got == set(true_atoms[row].tolist())
        assert float(result.fve[row]) > 0.999
        # coefficients match (order-insensitive)
        for atom, coeff in zip(true_atoms[row].tolist(), true_coeffs[row].tolist(), strict=True):
            slot = (result.atoms[row] == atom).nonzero().item()
            assert abs(float(result.coeffs[row][slot]) - coeff) < 0.02


def test_coefficients_nonnegative_and_fve_bounded() -> None:
    dictionary = _unit_dictionary()
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(16, 64, generator=gen)
    result = nnomp_batch(x, dictionary, max_atoms=16)
    assert float(result.coeffs.min()) >= 0.0
    assert float(result.fve.min()) >= -1e-5 and float(result.fve.max()) <= 1.0 + 1e-5


def test_masked_true_atom_never_selected() -> None:
    dictionary = _unit_dictionary()
    x = 2.0 * dictionary[42].unsqueeze(0)
    mask = torch.ones(1, dictionary.shape[0], dtype=torch.bool)
    mask[0, 42] = False
    result = nnomp_batch(x, dictionary, max_atoms=8, mask=mask)
    assert 42 not in set(result.atoms[0].tolist())


def test_stops_when_gain_stalls() -> None:
    dictionary = _unit_dictionary()
    x = 3.0 * dictionary[10].unsqueeze(0)  # one atom explains everything
    result = nnomp_batch(x, dictionary, max_atoms=16)
    used = int((result.atoms[0] >= 0).sum())
    assert used <= 2  # first atom -> FVE ~1; gain stalls immediately after
    assert float(result.fve[0]) > 0.999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="device-mismatch streaming needs CUDA")
def test_streamed_cpu_dictionary_matches_gpu_dictionary() -> None:
    """r2sp: a host-RAM dict (dict device != x device) must stream chunks and match exactly."""
    dictionary = _unit_dictionary()
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(8, 64, generator=gen).cuda()
    ref = nnomp_batch(x, dictionary.cuda(), max_atoms=6, chunk=100)  # same-device path
    got = nnomp_batch(x, dictionary, max_atoms=6, chunk=100)  # CPU dict -> streamed path
    assert torch.equal(got.atoms.cpu(), ref.atoms.cpu())
    assert torch.allclose(got.coeffs.cpu(), ref.coeffs.cpu(), atol=1e-4)
    assert torch.allclose(got.fve.cpu(), ref.fve.cpu(), atol=1e-5)


def test_half_mask_seeded_and_half() -> None:
    m1 = half_mask(64, 1000, seed=7)
    m2 = half_mask(64, 1000, seed=7)
    assert torch.equal(m1, m2)
    frac = float(m1.float().mean())
    assert 0.45 < frac < 0.55


def test_dictionary_sampling_and_dedup() -> None:
    mask = [False] * 3 + [True] * 40 + [False] * 2 + [True] * 10
    rendered = list(range(1000, 1055))
    shard = sample_dictionary_phrases(mask, rendered, max_positions=64, rng=random.Random(0))
    for n, phrases in shard.items():
        for phrase in phrases:
            assert len(phrase) == n
            start = rendered.index(phrase[0])
            assert all(mask[start : start + n])  # fully inside an assistant turn
    # long lengths cannot fit in the 10-token second turn beyond its bounds
    assert all(rendered.index(p[0]) + 32 <= 43 for p in shard.get(32, []))
    merged = merge_deduped([shard, shard])  # duplicate shard -> counts double, rows dedup
    for n, (ids, counts) in merged.items():
        assert ids.shape[0] == len(set(map(tuple, ids.tolist())))
        assert int(counts.min()) >= 2
        assert ids.shape[1] == n


def test_positions_manifest_replays_to_same_phrases() -> None:
    """The stage-3 capture replays the manifest: positions -> phrases must round-trip."""
    mask = [False] * 3 + [True] * 40 + [False] * 2 + [True] * 10
    rendered = list(range(1000, 1055))
    positions = sample_dictionary_positions(mask, rendered, max_positions=64, rng=random.Random(0))
    assert positions  # non-degenerate
    replayed = phrases_at_positions(mask, rendered, positions)
    direct = sample_dictionary_phrases(mask, rendered, max_positions=64, rng=random.Random(0))
    assert replayed == direct
    fits = fits_fn(mask, len(mask))
    for start in positions:
        assert start >= 1 and fits(start, 2)  # capture reads hidden[start - 1]
