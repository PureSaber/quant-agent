from pathlib import Path

from quant_agent.adapters.base import _read_csv_rows


def test_csv_size_limit(tmp_path: Path) -> None:
    big = tmp_path / "big.csv"
    big.write_text("a\n" + "1\n" * 100, encoding="utf-8")
    try:
        _read_csv_rows(big, max_read_mb=0.000001)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "limit" in str(exc).lower()
