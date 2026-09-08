import pytest

from tarf_rslv import TARFAccumulator


def _tarf(**overrides):
    defaults = dict(
        target_level=20.0,
        strike=100.0,
        fixing_dates=[0.5, 1.0],
        notional1=1.0,
        notional2=2.0,
    )
    defaults.update(overrides)
    return TARFAccumulator(**defaults)


def test_itm_fixing_accumulates_without_terminating():
    tarf = _tarf()
    outcome = tarf.settle(spot=105.0, previous_accumulated=0.0, fixing_index=0)

    assert outcome.cashflow == pytest.approx(5.0)
    assert outcome.accumulated == pytest.approx(5.0)
    assert outcome.terminated is False


def test_itm_fixing_terminates_without_adjustment_and_can_overshoot():
    tarf = _tarf(target_adjustment=0)
    outcome = tarf.settle(spot=105.0, previous_accumulated=18.0, fixing_index=0)

    assert outcome.terminated is True
    assert outcome.cashflow == pytest.approx(5.0)  # full intrinsic paid, overshooting the 20.0 target
    assert outcome.accumulated == pytest.approx(23.0)


def test_itm_fixing_terminates_with_strike_adjustment_caps_at_remaining_target():
    tarf = _tarf(target_adjustment=1)
    outcome = tarf.settle(spot=105.0, previous_accumulated=18.0, fixing_index=0)

    assert outcome.terminated is True
    assert outcome.cashflow == pytest.approx(2.0)  # exactly target_level - previous_accumulated


def test_itm_fixing_terminates_with_notional_adjustment_caps_at_remaining_target():
    tarf = _tarf(target_adjustment=2)
    outcome = tarf.settle(spot=105.0, previous_accumulated=18.0, fixing_index=0)

    assert outcome.terminated is True
    assert outcome.cashflow == pytest.approx(2.0)  # exactly target_level - previous_accumulated


def test_otm_fixing_pays_leveraged_loss_when_barrier_disabled():
    tarf = _tarf(barrier=0.0)
    outcome = tarf.settle(spot=95.0, previous_accumulated=0.0, fixing_index=0)

    assert outcome.cashflow == pytest.approx(-10.0)  # notional2 * (95 - 100)
    assert outcome.accumulated == pytest.approx(0.0)
    assert outcome.terminated is False


def test_otm_fixing_leveraged_loss_activates_beyond_the_barrier():
    tarf = _tarf(barrier=85.0)
    outcome = tarf.settle(spot=80.0, previous_accumulated=0.0, fixing_index=0)

    assert outcome.cashflow == pytest.approx(-40.0)  # notional2 * (80 - 100), KI has activated


def test_otm_fixing_leveraged_loss_is_suppressed_inside_the_barrier():
    tarf = _tarf(barrier=85.0)
    outcome = tarf.settle(spot=90.0, previous_accumulated=0.0, fixing_index=0)

    assert outcome.cashflow == pytest.approx(0.0)  # protected zone between barrier and strike


def test_barrier_check_has_no_memory_across_fixings():
    tarf = _tarf(barrier=85.0)
    breach = tarf.settle(spot=80.0, previous_accumulated=0.0, fixing_index=0)
    next_fixing = tarf.settle(spot=90.0, previous_accumulated=breach.accumulated, fixing_index=1)

    assert next_fixing.cashflow == pytest.approx(0.0)  # a prior breach does not carry forward


def test_inverted_target_uses_inverse_quote_increment():
    tarf = _tarf(inverted_target=True)
    outcome = tarf.settle(spot=105.0, previous_accumulated=0.0, fixing_index=0)

    expected_increment = (1.0 / 105.0 - 1.0 / 100.0) * -1.0
    assert outcome.accumulated == pytest.approx(expected_increment)
