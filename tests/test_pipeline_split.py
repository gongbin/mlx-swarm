import pytest

from swarm.pipeline import parse_split, split_layers


def counts(ranges):
    return [e - s for s, e in ranges]


def test_even_split_covers_all_layers_contiguously():
    ranges = parse_split("even", 28, 3)
    assert counts(ranges) == [10, 9, 9]
    assert ranges[0][0] == 0 and ranges[-1][1] == 28
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))


def test_weighted_split_is_proportional():
    assert counts(parse_split("w:20,12", 64, 2)) == [40, 24]
    assert counts(parse_split("w:1,3", 28, 2)) == [7, 21]


def test_every_rank_gets_at_least_one_layer():
    assert counts(split_layers(4, [100, 1, 1])) == [2, 1, 1]


def test_explicit_counts():
    assert parse_split("10,18", 28, 2) == [(0, 10), (10, 28)]


@pytest.mark.parametrize(
    "spec,layers,size",
    [("10,10", 28, 2), ("28,0", 28, 2), ("w:1,1,1", 28, 2), ("w:1,0", 28, 2)],
)
def test_invalid_specs_rejected(spec, layers, size):
    with pytest.raises(ValueError):
        parse_split(spec, layers, size)


def test_more_ranks_than_layers_rejected():
    with pytest.raises(ValueError):
        split_layers(2, [1, 1, 1])
