from adas.departure import DepartureConfig, DepartureWarning, State


def run(offsets, cfg=DepartureConfig()):
    w = DepartureWarning(cfg)
    return [w.update(o) for o in offsets]


def test_starts_lost_then_ok():
    assert run([0.0])[0] == State.OK


def test_single_spike_does_not_warn():
    s = run([0.0, 0.9, 0.0, 0.0])
    assert State.WARN_RIGHT not in s


def test_sustained_drift_warns_with_correct_side():
    assert run([0.0, 0.7, 0.8, 0.9])[-1] == State.WARN_RIGHT
    assert run([0.0, -0.7, -0.8, -0.9])[-1] == State.WARN_LEFT


def test_hysteresis_holds_between_clear_and_warn():
    s = run([0.7, 0.7, 0.7, 0.5, 0.5])        # 0.5 is below warn (0.6) but above clear (0.4)
    assert s[-1] == State.WARN_RIGHT
    assert run([0.7, 0.7, 0.7, 0.3])[-1] == State.OK


def test_lost_after_missing_frames_and_recovers():
    s = run([0.0] + [None] * 5 + [0.0])
    assert s[1] == State.OK and s[5] == State.LOST and s[6] == State.OK


def test_short_dropout_keeps_state():
    s = run([0.7, 0.7, 0.7, None, None])
    assert s[-1] == State.WARN_RIGHT
