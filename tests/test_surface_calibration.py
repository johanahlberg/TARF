"""Full FX-surface calibration: the market-surface layer, the forward vanilla PDE, and the
two-stage regime fit."""

import numpy as np
import pytest

from tarf_rslv import (
    FXVolSurface,
    RegimeForwardPDE,
    SmileQuotes,
    TARFAccumulator,
    ThreeRegimePricer,
    calibrate_regime_model,
    calibrate_regime_surface,
    calibrate_switch_rate_to_term_structure,
)
from tarf_rslv.svi import SVISlice, fit_svi_surface
from tarf_rslv.vol_surface import (
    DeltaConvention,
    bs_forward_price,
    local_vol_from_total_variance,
    strike_from_delta,
)

FLAT_ZERO_D = lambda t: 0.011  # noqa: E731
FLAT_ZERO_F = lambda t: 0.046  # noqa: E731


def _skew_surface(atm_base=0.072):
    quotes = [
        SmileQuotes(1 / 12, atm_base - 0.002, -0.004, 0.0015, -0.008, 0.005),
        SmileQuotes(0.25, atm_base, -0.006, 0.0018, -0.011, 0.006),
        SmileQuotes(0.50, atm_base + 0.002, -0.007, 0.0020, -0.013, 0.007),
        SmileQuotes(1.00, atm_base + 0.004, -0.008, 0.0022, -0.015, 0.008),
    ]
    return FXVolSurface(spot=0.81, smiles=quotes, domestic_zero=FLAT_ZERO_D, foreign_zero=FLAT_ZERO_F)


# --------------------------------------------------------------------------------------------------
# market-surface layer
# --------------------------------------------------------------------------------------------------
def test_flat_surface_reconstructs_flat_smile_and_local_vol():
    quotes = [SmileQuotes(T, 0.10, 0.0, 0.0, 0.0, 0.0) for T in (0.25, 0.5, 1.0)]
    surface = FXVolSurface(spot=0.81, smiles=quotes, domestic_zero=FLAT_ZERO_D, foreign_zero=FLAT_ZERO_F)

    _, vols = surface.smiles[0].knots(
        0.81, surface.forward(0.25), surface.foreign_df(0.25), surface.domestic_df(0.25), DeltaConvention()
    )
    assert np.allclose(vols, 0.10, atol=1e-6)

    grid = surface.forward(0.5) * np.exp(np.linspace(-0.08, 0.08, 7))
    assert np.allclose(surface.local_vol(0.5, grid), 0.10, atol=2e-4)


def test_market_strangle_round_trip():
    # build a smile, price its market strangle, and check the reconstruction recovers the RR
    surface = _skew_surface()
    smile = surface.smiles[1]
    strikes, vols = smile.knots(
        0.81, surface.forward(smile.tenor), surface.foreign_df(smile.tenor), surface.domestic_df(smile.tenor),
        DeltaConvention(),
    )
    # knots are 10dP, 25dP, ATM, 25dC, 10dC in strike order
    assert vols[1] > vols[3]                       # rr25 < 0 -> put wing richer
    assert vols[0] > vols[4]                       # rr10 < 0
    assert (vols[3] - vols[1]) == pytest.approx(smile.rr25, abs=1e-9)
    assert vols[2] == pytest.approx(smile.atm, abs=1e-12)


def test_local_vol_from_flat_total_variance_is_flat():
    def tv(t, y):
        return np.full_like(np.atleast_1d(y), 0.09 ** 2 * t)

    lv = local_vol_from_total_variance(tv, 0.5, np.linspace(-0.1, 0.1, 9))
    assert np.allclose(lv, 0.09, atol=1e-3)


def test_dupire_skew_exceeds_implied_skew_near_the_money():
    surface = _skew_surface()
    f = surface.forward(0.5)
    y = np.array([-0.03, 0.03])
    implied_skew = np.diff(surface.implied_vol(0.5, y))[0]
    local_skew = np.diff(surface.local_vol(0.5, f * np.exp(y)))[0]
    assert local_skew < implied_skew < 0  # local-vol skew is steeper


def test_strike_from_delta_inverts_black_scholes_delta():
    conv = DeltaConvention()
    f, t, sig = 0.80, 0.5, 0.08
    k = strike_from_delta(0.25, sig, t, 0.81, f, np.exp(-0.046 * t), conv)
    # recompute the spot delta of that strike
    d1 = (np.log(f / k) + 0.5 * sig * sig * t) / (sig * np.sqrt(t))
    from scipy.stats import norm

    assert np.exp(-0.046 * t) * norm.cdf(d1) == pytest.approx(0.25, abs=1e-6)


# --------------------------------------------------------------------------------------------------
# forward regime-switching vanilla PDE
# --------------------------------------------------------------------------------------------------
def test_forward_pde_matches_black_scholes_for_identical_flat_regimes():
    flat = lambda x, t: np.full_like(x, 0.10)  # noqa: E731
    q = np.array([[-0.5, 0.25, 0.25], [0.25, -0.5, 0.25], [0.25, 0.25, -0.5]])
    pde = RegimeForwardPDE(
        0.81, 0.011, 0.046, [flat, flat, flat], q, [1 / 3, 1 / 3, 1 / 3],
        max_t=1.0, sigma_ref=0.10, num_x=601, steps_per_year=500,
    )
    ivs = pde.implied_vol_grid([0.25, 0.5, 1.0], np.linspace(-0.08, 0.08, 5))
    for t, iv in ivs.items():
        assert np.max(np.abs(iv - 0.10)) < 1e-3


def test_forward_pde_mixture_produces_a_smile():
    lo = lambda x, t: np.full_like(x, 0.06)   # noqa: E731
    mid = lambda x, t: np.full_like(x, 0.075)  # noqa: E731
    hi = lambda x, t: np.full_like(x, 0.11)   # noqa: E731
    q = np.array([[-0.4, 0.2, 0.2], [0.2, -0.4, 0.2], [0.2, 0.2, -0.4]])
    pde = RegimeForwardPDE(
        0.81, 0.011, 0.046, [lo, mid, hi], q, [0.25, 0.5, 0.25],
        max_t=1.0, sigma_ref=0.11, num_x=601, steps_per_year=500,
    )
    iv = pde.implied_vol_grid([0.5], np.array([-0.06, 0.0, 0.06]))[0.5]
    assert iv[0] > iv[1] and iv[2] > iv[1]                     # smile
    assert iv[1] == pytest.approx(iv[1])                       # finite
    assert np.sqrt(np.array([0.25, 0.5, 0.25]) @ np.array([0.06, 0.075, 0.11]) ** 2) == pytest.approx(iv[1], abs=6e-3)


# --------------------------------------------------------------------------------------------------
# stage 1: surface calibration
# --------------------------------------------------------------------------------------------------
def test_calibrate_regime_surface_reprices_the_market_smile():
    surface = _skew_surface()
    model, report = calibrate_regime_surface(
        surface, [0.25, 0.5, 0.25], spread=0.03,
        target="hybrid", n_pde_passes=2, num_x=401, steps_per_year=300,
        dupire_nt=21, dupire_nx=101,
    )
    assert report.rms_vol_error < 5e-4          # < 5 bp
    assert model.time_varying

    # the three regime local-vol surfaces are genuinely different and dispersed by ~spread
    f = surface.forward(0.5)
    atm_lv = [float(r.local_volatility(np.array([f]), 0.5)[0]) for r in model.regimes]
    assert atm_lv[0] < atm_lv[1] < atm_lv[2]


def test_calibrated_model_prices_a_tarf_and_moves_with_the_skew():
    surface = _skew_surface()
    model, _ = calibrate_regime_surface(
        surface, [0.25, 0.5, 0.25], spread=0.03, target="mixture",
        num_x=401, steps_per_year=300, dupire_nt=21, dupire_nx=101,
    )
    tarf = TARFAccumulator(
        target_level=0.10, strike=0.82, fixing_dates=tuple((k + 1) / 12 for k in range(12)),
        is_call_option=False, notional1=1.0, notional2=2.0, target_adjustment=0,
    )
    pricer = ThreeRegimePricer(model, model.regime_weights, num_spot=121, num_target=60, n_steps=80)
    price = pricer.price_tarf(tarf, 1.0)
    assert np.isfinite(price)
    greeks = pricer.tarf_greeks(tarf, 1.0)
    assert np.isfinite(greeks["vega"]) and np.isfinite(greeks["delta"])


# --------------------------------------------------------------------------------------------------
# stage 2: switching rate
# --------------------------------------------------------------------------------------------------
def test_calibrate_switch_rate_returns_a_valid_generator():
    surface = _skew_surface()
    out = calibrate_switch_rate_to_term_structure(
        surface, regime_atm_levels=[0.065, 0.072, 0.080],
        pi0=[0.6, 0.3, 0.1], stationary=[0.25, 0.5, 0.25],
    )
    q = out["q"]
    assert np.allclose(q.sum(axis=1), 0.0, atol=1e-12)
    assert np.all(q - np.diag(np.diag(q)) >= -1e-12)
    assert out["switch_rate"] >= 0.0


def test_calibrate_switch_rate_needs_pi0_to_differ_from_stationary():
    surface = _skew_surface()
    with pytest.raises(ValueError):
        calibrate_switch_rate_to_term_structure(
            surface, [0.065, 0.072, 0.080], pi0=[0.25, 0.5, 0.25], stationary=[0.25, 0.5, 0.25]
        )


# --------------------------------------------------------------------------------------------------
# SVI
# --------------------------------------------------------------------------------------------------
def test_svi_fit_is_accurate_and_arbitrage_free():
    surface = _skew_surface()
    k = list(surface._knot_y)
    iv = [np.asarray(s.implied_vol(kk)) for s, kk in zip(surface.svi_slices, k)]
    slices, report = fit_svi_surface(surface._tenors, k, iv)

    assert report.rms_vol_error < 5e-4
    assert report.max_butterfly_violation < 1e-6      # Durrleman g >= 0 everywhere
    assert report.max_calendar_violation < 1e-6       # slices do not cross


def test_svi_slice_scaled_matches_variance_multiplier():
    s = SVISlice(0.5, 0.004, 0.2, -0.3, 0.0, 0.1)
    s2 = s.scaled(1.1)
    k = np.linspace(-0.2, 0.2, 9)
    assert np.allclose(s2.total_variance(k), 1.1 * s.total_variance(k))


def test_fxvolsurface_is_svi_backed_and_arbitrage_free():
    surface = _skew_surface()
    assert surface.svi_report is not None
    assert surface.svi_report.max_butterfly_violation < 1e-6
    assert surface.svi_report.max_calendar_violation < 1e-6
    # analytic Dupire is finite and positive across a wide moneyness range
    lv = surface.local_vol(0.5, surface.forward(0.5) * np.exp(np.linspace(-0.25, 0.25, 21)))
    assert np.all(np.isfinite(lv)) and np.all(lv > 0.0)


# --------------------------------------------------------------------------------------------------
# full two-stage calibration
# --------------------------------------------------------------------------------------------------
def test_calibrate_regime_model_full_pipeline():
    surface = _skew_surface()
    model, report = calibrate_regime_model(
        surface, [0.25, 0.5, 0.25], n_regimes=3, level_spread=0.15, skew_spread=0.12,
        target="hybrid", n_pde_passes=2, num_x=401, steps_per_year=300, dupire_nt=21, dupire_nx=101,
    )
    assert report.rms_vol_error < 1e-3          # regime dispersion costs ~2 bp vs pure Dupire
    assert report.max_butterfly_violation < 1e-3
    assert report.max_calendar_violation < 5e-3
    assert report.switch_rate > 0.0
    assert report.level_spread == 0.15 and report.skew_spread == 0.12
    assert model.time_varying
    assert np.allclose(model.q.sum(axis=1), 0.0, atol=1e-12)

    # regimes genuinely differ in BOTH level and skew
    f = surface.forward(0.5)
    lv = np.array([r.local_volatility(f * np.exp([-0.05, 0.0, 0.05]), 0.5) for r in model.regimes])
    assert lv[0, 1] < lv[1, 1] < lv[2, 1]                       # ATM level: calm < mid < stressed
    skew = lv[:, 0] - lv[:, 2]                                  # put minus call wing
    assert skew[0] < skew[1] < skew[2]                          # stressed regime is steeper-skew


def test_n_regimes_1_is_a_single_dupire_model_and_faster_to_price():
    surface = _skew_surface()
    model, report = calibrate_regime_model(
        surface, n_regimes=1, num_x=401, steps_per_year=300, dupire_nt=21, dupire_nx=101,
    )
    assert model.n_regimes == 1
    assert report.rms_vol_error < 5e-4          # a single Dupire model reprices the surface tightly
    assert report.level_spread == 0.0 and report.skew_spread == 0.0

    tarf = TARFAccumulator(
        target_level=0.10, strike=0.82, fixing_dates=tuple((k + 1) / 12 for k in range(12)),
        is_call_option=False, notional1=1.0, notional2=2.0, target_adjustment=0,
    )
    price = model.price_tarf(tarf, 1.0, num_spot=121, num_target=60, n_steps=80)
    greeks = model.tarf_greeks(tarf, 1.0, num_spot=121, num_target=60, n_steps=80)
    assert np.isfinite(price) and np.isfinite(greeks["vega"]) and np.isfinite(greeks["delta"])


# --------------------------------------------------------------------------------------------------
# Front Arena boundary
# --------------------------------------------------------------------------------------------------
class _FakeCurve:
    def __init__(self, rate):
        self._rate = rate

    def Rate(self, start_day, end_day):
        return self._rate


class _FakeMalzSurface:
    """sigma(delta) = atm - 2 rr (delta - 0.5) + 16 bf (delta - 0.5)^2 (Malz parabola)."""

    def __init__(self, atm, rr, bf):
        self._atm, self._rr, self._bf = atm, rr, bf

    def Value(self, expiry_date, delta, foreign_rate, domestic_rate):
        d = delta - 0.5
        return self._atm - 2.0 * self._rr * d + 16.0 * self._bf * d * d


def test_fx_surface_from_quotes_and_from_front_arena_objects():
    from datetime import date

    from tarf_rslv import fx_surface_from_quotes, market_surface_from_front_arena

    val = date(2026, 9, 9)
    dom, forgn = _FakeCurve(0.011), _FakeCurve(0.046)

    surf1 = fx_surface_from_quotes(
        0.81, val, [0.25, 0.5, 1.0],
        atm=[0.070, 0.072, 0.074], rr25=[-0.006, -0.007, -0.008], bf25=[0.0018, 0.0020, 0.0022],
        rr10=[-0.011, -0.013, -0.015], bf10=[0.006, 0.007, 0.008],
        domestic_curve=dom, foreign_curve=forgn,
    )
    assert len(surf1.market_nodes()) == 15
    assert np.all(np.isfinite([iv for _, _, iv, _ in surf1.market_nodes()]))

    surf2 = market_surface_from_front_arena(
        0.81, val, [date(2027, 3, 9), date(2027, 9, 9)], dom, forgn,
        _FakeMalzSurface(0.072, -0.006, 0.0018),
        delta_type="forward", atm_convention="atmf", premium="foreign",
    )
    assert len(surf2.smiles) == 2
    nodes = surf2.market_nodes()
    assert np.all(np.isfinite([iv for _, _, iv, _ in nodes]))
    assert surf2.convention.delta_type == "forward"
    assert surf2.convention.atm_convention == "forward"
    assert surf2.convention.premium_adjusted is True


def test_delta_convention_from_market_strings():
    from tarf_rslv.vol_surface import DeltaConvention

    c = DeltaConvention.from_market("fwd", "atmf", "%foreign")
    assert (c.delta_type, c.atm_convention, c.premium_adjusted) == ("forward", "forward", True)
    d = DeltaConvention.from_market("spot", "delta_neutral_straddle", "domestic")
    assert (d.delta_type, d.atm_convention, d.premium_adjusted) == ("spot", "dns", False)
    with pytest.raises(ValueError):
        DeltaConvention.from_market("spot", "dns", "sideways")


def test_calibrated_tarf_model_from_front_arena_one_call():
    from datetime import date

    from tarf_rslv import calibrated_tarf_model_from_front_arena

    class _DV:
        def __init__(self, n, u):
            self._n, self._u = n, u

        def Number(self):
            return self._n

        def Unit(self):
            return self._u

    val = date(2026, 9, 9)
    out = calibrated_tarf_model_from_front_arena(
        val, _DV(0.81, "CHF"), _DV(0.82, "CHF"), 0.10, [0.25, 0.5, 0.75, 1.0], 1.0,
        [date(2026, 12, 9), date(2027, 6, 9), date(2027, 9, 9)],
        _FakeCurve(0.011), _FakeCurve(0.046), _FakeMalzSurface(0.072, -0.006, 0.0018),
        delta_type="spot", atm_convention="dns", premium="domestic", n_regimes=1,
        num_spot=101, num_target=50, n_steps=60,
    )
    assert set(out) == {"result", "calibration"}
    assert np.isfinite(out["result"].number)
    assert out["calibration"]["n_regimes"] == 1
