from pathlib import Path


def test_llc_environment_is_separate_pinned_and_validated():
    requirements = Path("requirements-llc.txt").read_text()
    script = Path("scripts/prepare_llc_env.sh").read_text()
    assert "fbbf4c54e1f6ee46acb149f004df261fb05055c6" in requirements
    assert "llm-embeds-llc-v2" in script
    assert "--system-site-packages" in script
    assert "CUDA_VISIBLE_DEVICES=''" in script
    assert "validate_regular_logistic" in script
    assert "llc_environment.exit" in script
