from __future__ import annotations

def has_pot() -> bool:
    try:
        import ot  # type: ignore
        return True
    except Exception:
        return False