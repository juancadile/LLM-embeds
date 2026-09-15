from pathlib import Path


def test_report_queue_runs_after_terminal_llc_even_if_rejected():
    text = Path("scripts/continue_report.sh").read_text()
    assert "while [[ ! -f llc.exit ]]" in text
    assert "[[ $(<llc.exit) == 0 ]]" not in text
    assert "PYTHONPATH=analysis_v9" in text
    assert "--stage report" in text
    assert "report.exit" in text
