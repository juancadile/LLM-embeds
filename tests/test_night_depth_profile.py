from pathlib import Path


def test_depth_profile_is_layer_major_resumable_and_versioned():
    text = Path("scripts/night_depth_profile.sh").read_text()
    assert "PYTHONPATH=analysis_v9" in text
    assert "--stage mdl" in text
    assert "PREDICATES=(believes justified true lucky_guessed)" in text
    # layer-major: an interrupted run leaves every predicate at the same depths
    assert text.index('for layer in "${LAYERS[@]}"') < text.index('for predicate in "${PREDICATES[@]}"')
    assert '[[ -f $marker && $(<"$marker") == 0 ]] && continue' in text
    assert "depth_profile.exit" in text
