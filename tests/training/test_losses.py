"""Per-question losses: each question is its own softmax or sigmoid, never normalised across questions."""

import pytest
import torch

from selfjev.training.losses import grouped_loss

ITEMS = [
    {"type": "multiclass", "ids": [[1]] * 3, "target": 2},
    {"type": "binary", "ids": [[1]], "target": True},
    {"type": "multilabel", "ids": [[1]] * 4, "target": [True, False, False, True]},
    {"type": "multiclass", "ids": [[1]] * 2, "target": 0},
]


def test_grouped_loss_matches_per_question_definitions():
    s = torch.tensor([0.5, -1.0, 2.0, 0.3, 1.0, -2.0, 0.0, 3.0, -0.5, 0.7])
    F = torch.nn.functional
    want = (
        F.cross_entropy(s[None, 0:3], torch.tensor([2]))
        + F.binary_cross_entropy_with_logits(s[3:4], torch.tensor([1.0]))
        + F.binary_cross_entropy_with_logits(s[4:8], torch.tensor([1.0, 0, 0, 1]))
        + F.cross_entropy(s[None, 8:10], torch.tensor([0]))
    )
    assert torch.allclose(grouped_loss(s, ITEMS), want)


def test_no_normalisation_across_questions():
    """Changing another question's scores must not change a multiclass question's loss (no cross-question softmax)."""
    s1 = torch.tensor([0.5, -1.0, 2.0, 0.3, 1.0, -2.0, 0.0, 3.0, -0.5, 0.7])
    s2 = s1.clone()
    s2[3:10] += 50.0
    one = lambda s: grouped_loss(s[0:3], ITEMS[:1])
    assert one(s1) == one(s2)
    with pytest.raises(AssertionError):
        grouped_loss(s1[:9], ITEMS)  # misaligned scores are rejected, never silently padded
