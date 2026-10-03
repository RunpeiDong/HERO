"""Closed-loop contract of the HERO clip sampler.

The failure EMA counts raw failures, which accumulate in proportion to how often a clip is drawn. Static checks (a zero
table, one hand-set failing clip) cannot see what the sampler converges to once its own draws feed the table; these
tests iterate that loop to its deterministic steady state and check that the documented source shares survive it."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party/holosoma")]
from hero_isaacsim.managers.command.sampler import HeroAdaptiveTimestepsSampler, build_clip_prior

KW = {"adaptive_clip_max_probability": 1.0, "clip_cap_relative": 10.0}

# Source structure of a realistic mixed corpus (29,548 clips over 32 sources; anonymised): (clips, episode weight in units of 1/1420). The largest
# source holds 16,112 clips at a 14 % share, the smallest a single clip at 0.35 %; the fourth-largest source (8,000
# generated reaching clips) is weighted 20 %.
CORPUS_SOURCES = [
    (42, 5), (87, 5), (1, 1), (62, 15), (3, 5), (74, 25), (775, 40), (41, 30), (1205, 150), (1, 5), (38, 5), (8, 5),
    (92, 10), (251, 70), (234, 10), (1175, 200), (8000, 284), (16112, 200), (5, 10), (117, 80), (4, 5), (78, 5),
    (46, 5), (68, 50), (54, 10), (24, 10), (49, 20), (5, 20), (469, 70), (7, 10), (343, 30), (78, 30),
]


def _sampler(counts, weights, uniform=0.3):
    ids = torch.cat([torch.full((int(n),), i, dtype=torch.long) for i, n in enumerate(counts)])
    tags = [f"s{i}" for i in range(len(counts))]
    prior = build_clip_prior(ids, tags, dict(zip(tags, weights)), "cpu")
    n = int(ids.numel())
    return HeroAdaptiveTimestepsSampler(
        n * 100, "cpu", 50, per_clip=True, num_clips=n, max_clip_time_step=100, adaptive_uniform_ratio=uniform,
        clip_prior=prior, source_tag_ids=ids, source_tags=tags, **KW)


def _steady_state(s, difficulty, iters=300, beta=0.25):
    """Fixed point of the failure feedback: F_i is proportional to q_i * difficulty_i (episodes are drawn from q and a
    share difficulty_i of them fails), i.e. the steady state of the per-step failure EMA."""
    f = None
    for _ in range(iters):
        g = s.clip_marginal().double() * difficulty
        g = g / g.sum()
        f = g if f is None else (1.0 - beta) * f + beta * g
        s.bin_failed_count = (f[:, None] / s.num_bins).float().repeat(1, s.num_bins)
    return s.clip_marginal().double()


def _effective_clips(q):
    q = q[q > 0].double()
    return math.exp(float(-(q * q.log()).sum()))


def _source_frac(s, q):
    return torch.zeros(int(s.source_tag_ids.max()) + 1, dtype=torch.float64).index_add_(0, s.source_tag_ids, q)


def test_equal_difficulty_keeps_the_documented_mix():
    """Every clip equally hard -> the steady state must be the prior (source shares AND effective clip count)."""
    s = _sampler([2000, 60, 3], [0.5, 0.3, 0.2])  # per-clip shares 2.5e-4 / 5e-3 / 6.7e-2
    q = _steady_state(s, torch.ones(s.num_clips, dtype=torch.float64))
    frac = _source_frac(s, q)
    assert torch.allclose(frac, torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64), atol=0.01), frac.tolist()
    assert _effective_clips(q) >= 0.95 * _effective_clips(s.clip_prior), (_effective_clips(q), _effective_clips(s.clip_prior))


def test_a_mixed_corpus_does_not_concentrate_under_feedback():
    """With the 32-source structure and equal difficulty the steady state keeps >= 90 % of the prior's effective clip
    count and every source within 10 % of its weight. (Multiplying raw failure counts by the prior instead settles at a
    few hundred effective clips with 95 % of the draws on the small object sources.)"""
    counts = [c for c, _ in CORPUS_SOURCES]
    weights = [w for _, w in CORPUS_SOURCES]
    s = _sampler(counts, weights)
    q = _steady_state(s, torch.ones(s.num_clips, dtype=torch.float64), iters=200)
    w = torch.tensor(weights, dtype=torch.float64) / sum(weights)
    ratio = _source_frac(s, q) / w
    assert _effective_clips(q) >= 0.9 * _effective_clips(s.clip_prior), (_effective_clips(q), _effective_clips(s.clip_prior))
    assert float(ratio.min()) >= 0.9 and float(ratio.max()) <= 1.1, (float(ratio.min()), float(ratio.max()))


def test_failures_reweight_inside_a_source_but_never_take_it_over():
    """Hard clips are drawn more often than easy ones of the same source, bounded by clip_cap_relative x prior; a hard
    small source keeps its documented share instead of absorbing the sampler."""
    s = _sampler([2000, 60, 3], [0.5, 0.3, 0.2])
    difficulty = torch.full((s.num_clips,), 0.2, dtype=torch.float64)
    difficulty[:20] = 1.0          # 20 hard clips inside the big source
    difficulty[2060:] = 1.0        # the whole small source is hard
    q = _steady_state(s, difficulty)
    prior = s.clip_prior.double()
    frac = _source_frac(s, q)
    assert float(frac[2]) <= 0.2 * 1.05, frac.tolist()
    assert float((q[:20] / prior[:20]).min()) > 1.5                        # the curriculum still acts
    assert float((q / prior).max()) <= 10.0 * (1 + 1e-4)                    # and stays bounded
    assert _effective_clips(q) >= 0.5 * _effective_clips(prior), (_effective_clips(q), _effective_clips(prior))


def test_source_mass_is_preserved_exactly_under_the_relative_cap():
    """The cap is applied by water-filling inside each source, so even with every failure on one clip of a source the
    source keeps its share to float precision and the capped clip sits exactly at 10x its prior share."""
    s = _sampler([300, 200, 4], [0.6, 0.3, 0.1])
    s.bin_failed_count[0, :] = 1000.0      # one clip of the first source takes every failure
    s.bin_failed_count[500, :] = 1000.0    # and one clip of the 4-clip source
    q = s.clip_marginal().double()
    prior = s.clip_prior.double()
    frac = _source_frac(s, q)
    assert torch.allclose(frac, torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64), atol=1e-6), frac.tolist()
    assert abs(float(q[0] / prior[0]) - 10.0) < 1e-4
    # the 4-clip source cannot hold 10x: the capped clip takes (1-u) + u/4 of the source, the others share the rest
    assert float(q[500] / prior[500]) <= 10.0 * (1 + 1e-6) and float(q[500]) < 0.1
    assert abs(float(q.sum()) - 1.0) < 1e-6
    # a zero-weight source stays at exactly zero whatever its failures
    z = _sampler([10, 10], [1.0, 0.0])
    z.bin_failed_count[15, :] = 1e6
    assert float(z.clip_marginal()[10:].sum()) == 0.0


def test_the_relative_cap_holds_without_a_uniform_share():
    """``adaptive_uniform_ratio = 0``: a clip that never failed carries no mass of its own, so the within-source
    water-filling has nothing to spill onto unless the sampler floors the prior support. The cap must hold (the 10x
    bound used to be silently violated by up to n_s / m), the excess goes to the other clips of the same source and the
    source shares survive."""
    s = _sampler([100, 100], [0.5, 0.5], uniform=0.0)
    prior = s.clip_prior.double()
    s.bin_failed_count[0, :] = 5.0        # one failing clip in the first source ...
    s.bin_failed_count[100:, :] = 1.0     # ... every clip of the second
    q = s.clip_marginal().double()
    assert 9.99 * float(prior[0]) <= float(q[0]) <= 10.0 * float(prior[0]) * (1 + 1e-6), (float(q[0]), float(prior[0]))
    assert torch.allclose(_source_frac(s, q), torch.tensor([0.5, 0.5], dtype=torch.float64), atol=1e-6)
    assert float(q[1]) > 0.0 and abs(float(q.sum()) - 1.0) < 1e-6
    # one failing clip of 100 with no other failures in the corpus
    s.bin_failed_count.zero_()
    s.bin_failed_count[7, :] = 3.0
    q = s.clip_marginal().double()
    assert float(q[7]) <= 10.0 * float(prior[7]) * (1 + 1e-6)
    assert float(q[:100].sum()) == pytest.approx(0.5, abs=1e-6) and float(q[0]) == pytest.approx(0.5 / 99 * 0.9, rel=1e-3)
    # the projection refuses a support whose caps cannot hold the mass instead of renormalising past the caps
    with pytest.raises(ValueError, match="infeasible"):
        HeroAdaptiveTimestepsSampler._project_with_vector_cap(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
                                                              torch.tensor([0.3, 0.5, 0.5], dtype=torch.float64))
    # with the default uniform share the floor is inert: same table, same result as before
    t = _sampler([100, 100], [0.5, 0.5])
    t.bin_failed_count[0, :] = 5.0
    assert float(t.clip_marginal()[0]) == pytest.approx(10.0 * float(t.clip_prior[0]), rel=1e-5)
