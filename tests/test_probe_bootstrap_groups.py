import numpy as np

from src.probe_bootstrap import paired_indices, paired_resample
from src.probe_data import assign_splits, generate_scenarios, independent_groups


def _frame():
    return assign_splits(generate_scenarios(32, 3, 4), {"tune": .1, "coding": .7, "evaluation": .2}, 8)


def test_every_sampled_copy_enters_the_coder_as_its_own_group():
    """A group drawn twice must not collapse into one group: that shifts every endpoint."""
    frame = _frame()
    original = independent_groups(frame).to_numpy()
    split = frame.split.to_numpy()
    for seed in range(5):
        indices, copies = paired_resample(frame, seed)
        assert len(indices) == len(copies)
        for name in np.unique(split):
            mask = split[indices] == name
            sampled_instances = len(np.unique(original[split == name]))
            assert len(np.unique(copies[mask])) == sampled_instances
            # Rows of one copy all come from one original group.
            for label in np.unique(copies[mask]):
                assert len(np.unique(original[indices[mask][copies[mask] == label]])) == 1


def test_resampling_with_replacement_does_repeat_groups_so_the_fix_is_load_bearing():
    frame = _frame()
    original = independent_groups(frame).to_numpy()
    indices, copies = paired_resample(frame, 3)
    assert len(np.unique(copies)) > len(np.unique(original[indices]))


def test_paired_indices_is_unchanged_by_the_extension():
    frame = _frame()
    np.testing.assert_array_equal(paired_indices(frame, 11), paired_resample(frame, 11)[0])
