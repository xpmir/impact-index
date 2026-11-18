import pytest

import impact_index


def test_copy_mode_roundtrip():
    previous = impact_index.copy_mode()
    try:
        for mode in ("buffered", "native", "auto"):
            impact_index.set_copy_mode(mode)
            assert impact_index.copy_mode() == mode
        impact_index.set_copy_mode("Buffered")
        assert impact_index.copy_mode() == "buffered"
    finally:
        impact_index.set_copy_mode(previous)


def test_copy_mode_invalid():
    with pytest.raises(ValueError, match="unknown copy mode"):
        impact_index.set_copy_mode("fast")
