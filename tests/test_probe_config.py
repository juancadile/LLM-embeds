import pytest

from src.probe_data import PREDICATES
from src.probe_experiments import validate_predicates
from src.probe_utils import load_config


def test_full_config_predicates_are_strings_and_complete():
    cfg = load_config("configs/knows_mdl.yaml")
    assert cfg["predicates"] == list(PREDICATES)
    assert all(isinstance(predicate, str) for predicate in cfg["predicates"])


@pytest.mark.parametrize("predicates", [
    ["knows", True],
    ["knows", "unsupported"],
    ["knows", "knows"],
])
def test_predicate_validation_rejects_coercions_unknowns_and_duplicates(predicates):
    with pytest.raises(ValueError):
        validate_predicates({"predicates": predicates})
