from pathlib import Path


def test_depth_chain_waits_for_a_terminal_marker_and_preserves_it():
    text = Path("scripts/night_depth_chain.sh").read_text()
    assert "while [[ ! -f depth_profile.exit ]]" in text
    assert "kill -0" in text
    assert "mv -f depth_profile.exit depth_profile.slice1.exit" in text
    assert "exec bash night_depth_profile.sh" in text
