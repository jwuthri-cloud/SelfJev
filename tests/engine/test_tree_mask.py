"""The tree attention mask: ancestors plus the own causal prefix; the branch-only mask equals a slice of the full one."""

import pytest
import torch

from selfjev.engine.tree import build_tree, leaf_paths, tree_mask


def test_tree_mask_is_ancestors_plus_own_causal_prefix():
    t = build_tree([1, 2], [([3], [[4, 5], [6]]), ([7], [[8]])])
    m = tree_mask(t)
    # ids:  1 2 | 3 | 4 5 | 6 | 7 | 8 ; leaf "6" sees root + question 3 + itself, never 4 5 or the other question
    assert m[5].tolist() == [True, True, True, False, False, True, False, False]
    assert m[7].tolist() == [True, True, False, False, False, False, True, True]
    assert t["pos"] == [0, 1, 2, 3, 4, 3, 2, 3] and t["leaves"] == [4, 5, 7]
    assert leaf_paths(t) == [[0, 1, 2, 3, 4], [0, 1, 2, 5], [0, 1, 6, 7]]


@pytest.mark.parametrize("root_length", [1, 17, 513])
def test_branch_mask_equals_full_mask_slice(root_length):
    t = build_tree([1] * root_length, [([2, 3], [[4], [5, 6]]), ([7], [[8, 9, 10]])])
    full = tree_mask(t)
    branch = tree_mask(t, branches_only=True)
    assert torch.equal(branch, full[root_length:, root_length:])
    assert full[root_length:, :root_length].all()


def test_branch_mask_size_does_not_grow_with_document():
    t = build_tree([1] * 32000, [([2], [[3], [4]])])
    assert tree_mask(t, branches_only=True).shape == (3, 3)
