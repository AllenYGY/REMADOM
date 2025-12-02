from __future__ import annotations

from remadom.utils.schedules import build_schedule


def test_linear_schedule():
    start, fn = build_schedule({"kind": "linear", "start": 0.0, "end": 1.0, "epochs": 4}, default=0.0, epochs=4)
    assert start == 0.0
    assert fn is not None
    vals = [fn(t) for t in range(4)]
    assert vals[0] == 0.0
    assert vals[-1] == 1.0


def test_step_schedule():
    start, fn = build_schedule({"kind": "step", "start": 1.0, "gamma": 0.5, "step_size": 2}, default=1.0, epochs=6)
    assert start == 1.0
    assert fn is not None
    v0 = fn(0)
    v1 = fn(2)
    v2 = fn(4)
    assert v1 == v0 * 0.5
    assert v2 == v1 * 0.5


def test_cosine_restart_schedule():
    start, fn = build_schedule({"kind": "cosine_restart", "start": 1.0, "end": 0.0, "epochs": 4, "cycles": 2}, default=1.0, epochs=4)
    assert start == 1.0
    assert fn is not None
    vals = [fn(t) for t in range(4)]
    assert max(vals) <= 1.0
    assert min(vals) >= 0.0
