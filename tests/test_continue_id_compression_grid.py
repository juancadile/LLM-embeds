from pathlib import Path


def test_grid_queue_is_detached_safe_gated_and_resumable():
    text = Path("scripts/continue_id_compression_grid.sh").read_text()
    assert "while [[ ! -f quality_gates.exit ]]" in text
    assert "$(<quality_gates.exit) != 0" in text
    assert "while [[ ! -f compression_primary_hotfix.exit ]]" in text
    assert "$(<compression_primary_hotfix.exit) != 0" in text
    assert "comparative_claim_eligible" in text
    assert "intrinsic_dimension.parquet" in text
    assert "compression.parquet" in text
    assert "complete_table" in text
    assert "id_compression_grid.exit" in text
    assert "CUDA_VISIBLE_DEVICES=0" in text
    assert "ensure_primary_block29_mdl" in text
    assert "layers+=(block29)" in text
