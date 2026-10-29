"""
Regression tests for the two NIFTY Lab corrections:

  1. Intraday targets are aligned by EXACT TIMESTAMP and never cross a
     trading-session boundary (zorix/nifty_lab/features.py).
  2. Historical one-month-ahead forecasting: user-selected target date,
     calendar-month information cutoff, labels fully observable at the
     cutoff, evaluation against the exact target session
     (zorix/nifty_lab/historical_sim.py).

Run from the project root (no Streamlit UI is launched; importing the
package only imports the streamlit module, it does not start an app):

    python -m pytest test_bug_fixes.py -v

All data below is deterministic and synthetic. Intraday prices are a pure
function of the bar's timestamp (`price_at`), so every expectation is
computed independently of row positions. The one real-data class at the end
reads the project's own CSV and is skipped if the file is absent.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zorix.nifty_lab.config import DEFAULT_NIFTYLAB_CONFIG
from zorix.nifty_lab.features import (
    assign_session_ids, build_intraday_features, build_intraday_targets,
)
from zorix.nifty_lab.historical_sim import (
    HistoricalForecastError, build_horizon_training_set, calendar_month_cutoff,
    run_historical_forecast,
)
from zorix.nifty_lab.intraday_model import train_intraday_models

PROJECT_ROOT = Path(__file__).resolve().parent

# Small/fast forest settings; hyper-parameters are irrelevant to what is tested.
FAST_CFG = dataclasses.replace(
    DEFAULT_NIFTYLAB_CONFIG,
    nextday_n_estimators=15, nextday_n_jobs=1,
    intraday_n_estimators=10, intraday_n_jobs=1,
)


# ======================================================================
# Intraday synthetic data helpers
# ======================================================================

BASE = pd.Timestamp("2024-01-01 00:00:00")


def price_at(ts) -> float:
    """Close of the synthetic bar stamped `ts`: a pure function of the timestamp
    (non-monotonic so RSI etc. are defined, unique per minute in practice)."""
    m = (pd.Timestamp(ts) - BASE).total_seconds() / 60.0
    return float(20000 + 10 * np.sin(m / 11.0) + 0.37 * np.cos(m / 3.0) + m * 1e-4)


def session(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="min")


def make_bars(index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.array([price_at(t) for t in index])
    return pd.DataFrame(
        {"Open": close - 0.1, "High": close + 0.3, "Low": close - 0.3, "Close": close},
        index=pd.DatetimeIndex(index, name="datetime"),
    )


def row(frame: pd.DataFrame, ts: str) -> pd.Series:
    return frame.loc[pd.Timestamp(ts)]


def assert_invalid(r: pd.Series, h: int) -> None:
    assert pd.isna(r[f"target_ts_{h}m"])
    assert np.isnan(r[f"target_close_{h}m"])
    assert np.isnan(r[f"fwd_ret_{h}m"])
    assert np.isnan(r[f"fwd_dir_{h}m"])


def assert_valid(r: pd.Series, origin_ts: str, h: int) -> None:
    """Valid target == exactly origin + h minutes, with that bar's own close."""
    expected_ts = pd.Timestamp(origin_ts) + pd.Timedelta(minutes=h)
    assert r[f"target_ts_{h}m"] == expected_ts
    assert r[f"target_close_{h}m"] == pytest.approx(price_at(expected_ts))
    assert r[f"fwd_ret_{h}m"] == pytest.approx(price_at(expected_ts) / price_at(origin_ts) - 1)
    assert r[f"fwd_dir_{h}m"] == float(price_at(expected_ts) > price_at(origin_ts))


# ======================================================================
# 1. Intraday target alignment
# ======================================================================

class TestIntradayTargetTimestamps:

    @pytest.mark.parametrize("h", [5, 15, 30])
    def test_contiguous_session_targets_are_exactly_t_plus_h(self, h):
        idx = session("2024-03-04 09:15", "2024-03-04 15:29")
        out = build_intraday_targets(make_bars(idx), [h])
        last = idx[-1]
        for ts in idx:
            r = out.loc[ts]
            if ts + pd.Timedelta(minutes=h) <= last:
                assert_valid(r, ts, h)
            else:  # target would be after the last bar of the session
                assert_invalid(r, h)
        assert out[f"target_ts_{h}m"].notna().sum() == len(idx) - h

    def test_all_three_horizons_in_one_call(self):
        out = build_intraday_targets(make_bars(session("2024-03-04 09:15", "2024-03-04 11:00")), [5, 15, 30])
        for h in (5, 15, 30):
            assert_valid(row(out, "2024-03-04 09:30"), "2024-03-04 09:30", h)
        # 10:45 + 30 = 11:15 > last bar 11:00 -> only the 30m target is invalid
        r = row(out, "2024-03-04 10:45")
        assert_valid(r, "2024-03-04 10:45", 5)
        assert_valid(r, "2024-03-04 10:45", 15)
        assert_invalid(r, 30)

    def test_missing_target_bar_is_invalid_not_substituted(self):
        idx = session("2024-03-04 09:15", "2024-03-04 11:00").delete(
            session("2024-03-04 09:15", "2024-03-04 11:00").get_loc(pd.Timestamp("2024-03-04 10:20")))
        out = build_intraday_targets(make_bars(idx), [5])
        # 10:15 + 5 = 10:20 is missing. A row offset would silently use 10:21.
        r = row(out, "2024-03-04 10:15")
        assert_invalid(r, 5)
        assert r["target_close_5m"] != price_at("2024-03-04 10:21")
        # neighbours whose own target bar exists are unaffected
        assert_valid(row(out, "2024-03-04 10:14"), "2024-03-04 10:14", 5)  # -> 10:19
        assert_valid(row(out, "2024-03-04 10:16"), "2024-03-04 10:16", 5)  # -> 10:21

    def test_missing_intermediate_bar_does_not_shift_the_target(self):
        full = session("2024-03-04 09:15", "2024-03-04 11:00")
        idx = full.delete(full.get_loc(pd.Timestamp("2024-03-04 10:17")))
        out = build_intraday_targets(make_bars(idx), [5])
        # 10:15 + 5 = 10:20 exists; with 10:17 absent, "5 rows ahead" would be 10:21.
        r = row(out, "2024-03-04 10:15")
        assert_valid(r, "2024-03-04 10:15", 5)
        assert r["target_close_5m"] != price_at("2024-03-04 10:21")

    def test_overnight_gap_never_crossed(self):
        idx = session("2024-03-04 09:15", "2024-03-04 15:29").append(
            session("2024-03-05 09:15", "2024-03-05 15:29"))
        out = build_intraday_targets(make_bars(idx), [5, 30, 1066])
        # last bars of day 1: target time would be in the overnight gap / next morning
        for ts in ("2024-03-04 15:25", "2024-03-04 15:29"):
            assert_invalid(row(out, ts), 5)
            assert_invalid(row(out, ts), 30)
        # 15:29 + 1066 min == 2024-03-05 09:15, a bar that EXISTS -- but in the next session
        assert pd.Timestamp("2024-03-05 09:15") in idx
        assert_invalid(row(out, "2024-03-04 15:29"), 1066)
        # first bars of day 2 are valid and stay inside day 2
        assert_valid(row(out, "2024-03-05 09:15"), "2024-03-05 09:15", 30)

    def test_month_end_overnight_gap_never_crossed(self):
        idx = session("2024-01-31 15:20", "2024-01-31 15:29").append(
            session("2024-02-01 09:15", "2024-02-01 09:25"))
        out = build_intraday_targets(make_bars(idx), [5])
        assert_invalid(row(out, "2024-01-31 15:29"), 5)
        assert_valid(row(out, "2024-02-01 09:15"), "2024-02-01 09:15", 5)

    def test_sixteen_minute_same_date_break_is_a_session_boundary(self):
        """Mirrors the 2023-09-15 break in the project data (15:45 -> 16:01)."""
        idx = session("2024-09-16 09:15", "2024-09-16 15:45").append(session("2024-09-16 16:01", "2024-09-16 16:30"))
        out = build_intraday_targets(make_bars(idx), [5, 16])
        assert pd.Timestamp("2024-09-16 16:01") in idx
        assert_invalid(row(out, "2024-09-16 15:45"), 16)      # 16:01 exists, other session
        assert_invalid(row(out, "2024-09-16 15:44"), 5)       # 15:49: inside the break
        assert_valid(row(out, "2024-09-16 15:40"), "2024-09-16 15:40", 5)
        assert_valid(row(out, "2024-09-16 16:01"), "2024-09-16 16:01", 5)

    def test_same_date_session_break_with_existing_target_bar_is_invalid(self):
        """The decisive case: the target timestamp EXISTS but belongs to the
        next session of the same date. Date-only logic accepts it."""
        s1 = session("2024-03-04 09:15", "2024-03-04 11:30")
        s2 = session("2024-03-04 13:00", "2024-03-04 15:29")
        idx = s1.append(s2)
        assert pd.Timestamp("2024-03-04 13:00") in idx
        out = build_intraday_targets(make_bars(idx), [90])
        assert_invalid(row(out, "2024-03-04 11:30"), 90)       # 11:30 + 90 = 13:00, other session
        out5 = build_intraday_targets(make_bars(idx), [5])
        assert_valid(row(out5, "2024-03-04 11:20"), "2024-03-04 11:20", 5)   # within session 1
        assert_invalid(row(out5, "2024-03-04 11:28"), 5)                      # 11:33 is in the break
        assert_valid(row(out5, "2024-03-04 13:10"), "2024-03-04 13:10", 5)   # within session 2

    def test_small_gap_is_missing_bars_not_a_new_session(self):
        # Bars 10:01-10:03 are missing (a 4-minute hole, within the 5-minute tolerance).
        full = session("2024-03-04 09:15", "2024-03-04 10:30")
        idx = full.delete([full.get_loc(pd.Timestamp(t)) for t in
                           ("2024-03-04 10:01", "2024-03-04 10:02", "2024-03-04 10:03")])
        out = build_intraday_targets(make_bars(idx), [5])
        assert_invalid(row(out, "2024-03-04 09:56"), 5)                         # -> 10:01, missing
        assert_invalid(row(out, "2024-03-04 09:58"), 5)                         # -> 10:03, missing
        assert_valid(row(out, "2024-03-04 09:59"), "2024-03-04 09:59", 5)       # -> 10:04, same session
        assert_valid(row(out, "2024-03-04 10:00"), "2024-03-04 10:00", 5)       # -> 10:05

    def test_session_ids_split_on_gap_larger_than_threshold(self):
        idx = session("2024-03-04 09:15", "2024-03-04 09:20").append(
            session("2024-03-04 09:26", "2024-03-04 09:30"))   # 6-minute gap
        ids = assign_session_ids(idx, session_gap_minutes=5)
        assert list(ids) == [0] * 6 + [1] * 5
        ids_loose = assign_session_ids(idx, session_gap_minutes=10)
        assert set(ids_loose) == {0}
        assert len(assign_session_ids(pd.DatetimeIndex([]))) == 0

    def test_explicit_session_id_column_overrides_time_gaps(self):
        idx = session("2024-03-04 09:15", "2024-03-04 10:00")   # perfectly contiguous
        bars = make_bars(idx)
        bars["session_id"] = np.where(idx < pd.Timestamp("2024-03-04 09:40"), "A", "B")
        out = build_intraday_targets(bars, [5])
        assert_invalid(row(out, "2024-03-04 09:38"), 5)   # 09:43 exists but is session B
        assert_valid(row(out, "2024-03-04 09:30"), "2024-03-04 09:30", 5)
        assert_valid(row(out, "2024-03-04 09:45"), "2024-03-04 09:45", 5)

    def test_invalid_direction_is_nan_not_zero(self):
        out = build_intraday_targets(make_bars(session("2024-03-04 09:15", "2024-03-04 09:40")), [5])
        tail = out.iloc[-5:]
        assert tail["fwd_dir_5m"].isna().all()
        assert tail["fwd_ret_5m"].isna().all()
        assert out.iloc[:-5]["fwd_dir_5m"].isin([0.0, 1.0]).all()

    def test_price_return_and_timestamp_describe_the_same_bar(self):
        idx = session("2024-03-04 09:15", "2024-03-04 11:30").append(session("2024-03-04 13:00", "2024-03-04 14:00"))
        bars = make_bars(idx)
        out = build_intraday_targets(bars, [5, 15, 30])
        for h in (5, 15, 30):
            ok = out[f"target_ts_{h}m"].notna()
            assert ok.any()
            looked_up = bars["Close"].loc[out.loc[ok, f"target_ts_{h}m"]].to_numpy()
            np.testing.assert_allclose(out.loc[ok, f"target_close_{h}m"].to_numpy(), looked_up)
            np.testing.assert_allclose(out.loc[ok, f"fwd_ret_{h}m"].to_numpy(),
                                       looked_up / out.loc[ok, "Close"].to_numpy() - 1)
            delta = out.loc[ok, f"target_ts_{h}m"] - out.loc[ok].index.to_series()
            assert (delta == pd.Timedelta(minutes=h)).all()

    def test_duplicate_or_unsorted_index_is_rejected(self):
        idx = session("2024-03-04 09:15", "2024-03-04 09:30")
        with pytest.raises(ValueError):
            build_intraday_targets(make_bars(idx.append(idx[:1])), [5])
        with pytest.raises(ValueError):
            build_intraday_targets(make_bars(idx[::-1]), [5])

    def test_full_pipeline_with_features_uses_raw_bars_for_lookup(self):
        full = session("2024-03-04 09:15", "2024-03-04 12:00").append(
            session("2024-03-05 09:15", "2024-03-05 12:00"))
        raw_idx = full.delete([full.get_loc(pd.Timestamp(t)) for t in
                               ("2024-03-04 10:20", "2024-03-04 11:41", "2024-03-05 10:05")])
        raw = make_bars(raw_idx)
        feat = build_intraday_features(raw)
        assert len(feat) < len(raw)            # warm-up rows were dropped
        out = build_intraday_targets(feat, [5, 15, 30], bars=raw)
        raw_set = set(raw_idx)
        for h in (5, 15, 30):
            for ts, r in out.iterrows():
                tgt = ts + pd.Timedelta(minutes=h)
                if tgt in raw_set and tgt.date() == ts.date():
                    assert_valid(r, ts, h)
                else:
                    assert_invalid(r, h)


    def test_bars_that_exist_but_were_dropped_from_features_are_still_found(self):
        """Flat prices make RSI undefined, so build_intraday_features drops those
        rows. Their bars exist in the raw data and are legitimate targets."""
        noisy = make_bars(session("2024-03-04 09:15", "2024-03-04 10:00"))
        flat_idx = session("2024-03-04 10:01", "2024-03-04 10:40")
        flat = pd.DataFrame({c: float(noisy["Close"].iloc[-1]) for c in ("Open", "High", "Low", "Close")},
                            index=pd.DatetimeIndex(flat_idx, name="datetime"))
        raw = pd.concat([noisy, flat])
        feat = build_intraday_features(raw)
        origin = feat.index[-5]                      # last surviving rows of the feature frame...
        target = origin + pd.Timedelta(minutes=5)    # ...so this target is the first dropped bar
        assert target in raw.index and target not in feat.index      # dropped from features, present in raw
        assert origin in feat.index
        with_raw = build_intraday_targets(feat, [5], bars=raw)
        assert with_raw.loc[origin, "target_ts_5m"] == target
        assert with_raw.loc[origin, "target_close_5m"] == raw.loc[target, "Close"]
        # looking only inside the (reduced) feature frame cannot see that bar
        assert pd.isna(build_intraday_targets(feat, [5]).loc[origin, "target_ts_5m"])


class TestIntradayForecastObservation:

    def test_forecast_price_timestamp_and_origin_refer_to_same_bar(self):
        idx = session("2024-03-04 09:15", "2024-03-04 15:29")
        for d in range(5, 9):
            idx = idx.append(session(f"2024-03-{d:02d} 09:15", f"2024-03-{d:02d} 15:29"))
        bars = make_bars(idx.drop_duplicates())
        res = train_intraday_models(bars, FAST_CFG, train_window=800)
        assert set(res.horizons) == {5, 15, 30}
        origin = res.as_of
        assert origin == bars.index[-1]
        for h, hr in res.horizons.items():
            assert hr.forecast_origin == origin
            assert hr.target_timestamp == origin + pd.Timedelta(minutes=h)
            assert hr.origin_price == pytest.approx(price_at(origin))
            assert hr.last_forecast_price == pytest.approx(
                round(price_at(origin) * (1 + hr.last_forecast_ret), 2))
            assert hr.actual_price is None   # the target bar has not happened yet


# ======================================================================
# 2. Historical one-month-ahead forecast (synthetic daily data)
# ======================================================================

def make_daily(n: int = 600, seed: int = 7) -> pd.DataFrame:
    """Deterministic random-walk daily bars on business days, with a few
    holidays removed so that sessions != weekdays."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n + 10)
    holidays = pd.to_datetime(["2022-01-26", "2022-08-15", "2022-12-26", "2023-01-26", "2023-05-01"])
    dates = dates[~dates.isin(holidays)][:n]
    close = 20000 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close},
                        index=pd.DatetimeIndex(dates, name="datetime"))


@pytest.fixture(scope="module")
def daily() -> pd.DataFrame:
    return make_daily()


TARGET = pd.Timestamp("2023-09-14")   # a trading session well inside the data


class TestCalendarMonthCutoff:

    @pytest.mark.parametrize("target,expected", [
        ("2024-03-15", "2024-02-15"),
        ("2024-03-31", "2024-02-29"),   # leap-year month-end clamp
        ("2023-03-31", "2023-02-28"),
        ("2023-03-30", "2023-02-28"),
        ("2024-01-31", "2023-12-31"),   # across a year boundary
        ("2024-03-29", "2024-02-29"),
    ])
    def test_cutoff_is_one_calendar_month_before_target(self, target, expected):
        assert calendar_month_cutoff(target) == pd.Timestamp(expected)

    def test_result_reports_selected_target_and_cutoff(self, daily):
        res = run_historical_forecast(daily, FAST_CFG, TARGET)
        assert res.target_date == TARGET
        assert res.cutoff_date == pd.Timestamp("2023-08-14")
        assert res.last_session_used <= res.cutoff_date
        # the cutoff follows the TARGET, not the end of the dataset
        other = run_historical_forecast(daily, FAST_CFG, pd.Timestamp("2023-11-20"))
        assert other.cutoff_date == pd.Timestamp("2023-10-20")
        assert other.cutoff_date != res.cutoff_date

    def test_dataset_end_is_irrelevant(self, daily):
        full = run_historical_forecast(daily, FAST_CFG, TARGET)
        cut = run_historical_forecast(daily.loc[:TARGET], FAST_CFG, TARGET)
        assert cut.predicted_price == full.predicted_price
        assert cut.cutoff_date == full.cutoff_date

    def test_cutoff_on_a_trading_day_includes_that_session(self, daily):
        target = next(t for t in daily.index[300:] if calendar_month_cutoff(t) in daily.index)
        res = run_historical_forecast(daily, FAST_CFG, target)
        assert res.last_session_used == res.cutoff_date

    def test_cutoff_on_a_weekend_uses_previous_session(self, daily):
        target = next(t for t in daily.index[300:] if calendar_month_cutoff(t).weekday() >= 5)
        res = run_historical_forecast(daily, FAST_CFG, target)
        assert res.cutoff_date.weekday() >= 5
        assert res.last_session_used == daily.index[daily.index <= res.cutoff_date][-1]
        assert res.last_session_used < res.cutoff_date

    def test_horizon_counts_sessions_not_weekdays(self, daily):
        # window containing the 2023-05-01 holiday (a weekday that is not a session)
        target = pd.Timestamp("2023-05-31")
        res = run_historical_forecast(daily, FAST_CFG, target)
        pos = daily.index.get_loc
        assert res.horizon_sessions == pos(target) - pos(res.last_session_used)
        weekdays = len(pd.bdate_range(res.last_session_used + pd.Timedelta(days=1), target))
        assert res.horizon_sessions == weekdays - 1   # one holiday in the window


class TestTrainingLabelsObservableAtCutoff:

    @staticmethod
    def _training_set(daily, target):
        cutoff = calendar_month_cutoff(target)
        info = daily.loc[daily.index <= cutoff]
        h = daily.index.get_loc(target) - daily.index.get_loc(info.index[-1])
        return info, h, cutoff, build_horizon_training_set(info, h, cutoff)

    def test_no_label_outcome_is_after_the_cutoff(self, daily):
        info, h, cutoff, (X, y, feat_dates, label_dates, feat) = self._training_set(daily, TARGET)
        assert len(y) == len(feat_dates) == len(label_dates) > 50
        assert label_dates.max() <= cutoff
        assert feat_dates.max() <= cutoff

    def test_each_label_is_the_close_exactly_h_sessions_later(self, daily):
        info, h, cutoff, (X, y, feat_dates, label_dates, feat) = self._training_set(daily, TARGET)
        for fd, ld, label in zip(feat_dates, label_dates, y):
            assert daily.index.get_loc(ld) - daily.index.get_loc(fd) == h
            assert label == pytest.approx(daily.loc[ld, "Close"] / daily.loc[fd, "Close"] - 1)

    def test_rows_whose_outcome_was_not_yet_observed_are_excluded(self, daily):
        """The last h sessions before the cutoff have no observable h-ahead
        outcome, so they must not be training rows (this is what
        `train_end < test_start` alone fails to check)."""
        info, h, cutoff, (X, y, feat_dates, label_dates, feat) = self._training_set(daily, TARGET)
        last_session = info.index[-1]
        assert feat_dates.max() == info.index[info.index.get_loc(last_session) - h]
        assert feat.index[-1] == last_session      # present only for prediction
        assert last_session not in feat_dates

    def test_result_exposes_label_dates_not_after_cutoff(self, daily):
        res = run_historical_forecast(daily, FAST_CFG, TARGET)
        assert res.last_train_label_date <= res.cutoff_date
        assert res.last_train_feature_date < res.last_train_label_date
        assert res.n_train_rows > 50

    def test_tripwire_rejects_information_after_the_cutoff(self, daily):
        cutoff = calendar_month_cutoff(TARGET)
        with pytest.raises(RuntimeError, match="past the cutoff"):
            build_horizon_training_set(daily, 21, cutoff)   # full frame, not truncated

    def test_prediction_is_independent_of_every_post_cutoff_price(self, daily):
        base = run_historical_forecast(daily, FAST_CFG, TARGET)
        rng = np.random.default_rng(99)
        perturbed = daily.copy()
        later = perturbed.index > base.last_session_used
        perturbed.loc[later, ["Open", "High", "Low", "Close"]] *= rng.uniform(0.4, 2.5, (later.sum(), 1))
        redo = run_historical_forecast(perturbed, FAST_CFG, TARGET)
        assert redo.predicted_price == base.predicted_price          # bit-identical forecast
        assert redo.n_train_rows == base.n_train_rows
        assert redo.actual_price != base.actual_price                # the actual did change
        assert redo.forecast_error == pytest.approx(redo.predicted_price - redo.actual_price)

    def test_perturbation_check_has_power_pre_cutoff_prices_do_matter(self, daily):
        base = run_historical_forecast(daily, FAST_CFG, TARGET)
        changed = daily.copy()
        pre = changed.index <= base.last_session_used
        changed.loc[pre, ["Open", "High", "Low", "Close"]] *= np.linspace(0.8, 1.3, pre.sum())[:, None]
        assert run_historical_forecast(changed, FAST_CFG, TARGET).predicted_price != base.predicted_price


class TestForecastEvaluatedAgainstExactTarget:

    def test_actual_is_the_close_of_the_selected_session(self, daily):
        marked = daily.copy()
        marked.loc[TARGET, "Close"] = 12345.67          # make the target session unmistakable
        res = run_historical_forecast(marked, FAST_CFG, TARGET)
        assert res.actual_price == 12345.67
        prev_day, next_day = marked.index[marked.index.get_loc(TARGET) - 1], marked.index[marked.index.get_loc(TARGET) + 1]
        assert res.actual_price not in (marked.loc[prev_day, "Close"], marked.loc[next_day, "Close"])

    def test_error_fields_are_consistent(self, daily):
        res = run_historical_forecast(daily, FAST_CFG, TARGET)
        assert res.actual_price == daily.loc[TARGET, "Close"]
        assert res.forecast_error == pytest.approx(res.predicted_price - res.actual_price)
        assert res.abs_error == pytest.approx(abs(res.forecast_error))
        assert res.pct_error == pytest.approx(res.forecast_error / res.actual_price)
        assert res.baseline_price == daily.loc[res.last_session_used, "Close"]
        assert res.baseline_error == pytest.approx(res.baseline_price - res.actual_price)
        frame = res.to_frame()
        assert frame.loc[0, "Target date"] == TARGET.date()
        assert frame.loc[0, "Cutoff date"] == res.cutoff_date.date()
        assert frame.loc[0, "Predicted close"] == res.predicted_price

    def test_deterministic(self, daily):
        a = run_historical_forecast(daily, FAST_CFG, TARGET)
        b = run_historical_forecast(daily, FAST_CFG, TARGET)
        assert a.predicted_price == b.predicted_price

    def test_time_of_day_in_target_is_ignored(self, daily):
        res = run_historical_forecast(daily, FAST_CFG, pd.Timestamp("2023-09-14 15:30"))
        assert res.target_date == TARGET


class TestExplicitFailureModes:

    def test_weekend_is_a_non_trading_date(self, daily):
        with pytest.raises(HistoricalForecastError) as e:
            run_historical_forecast(daily, FAST_CFG, "2023-09-16")   # Saturday
        assert e.value.code == "NON_TRADING_DATE"
        assert "2023-09-15" in str(e.value) and "2023-09-18" in str(e.value)

    def test_holiday_inside_coverage_is_a_non_trading_date(self, daily):
        assert pd.Timestamp("2023-01-26") not in daily.index and pd.Timestamp("2023-01-26").weekday() < 5
        with pytest.raises(HistoricalForecastError) as e:
            run_historical_forecast(daily, FAST_CFG, "2023-01-26")
        assert e.value.code == "NON_TRADING_DATE"

    @pytest.mark.parametrize("date", ["2021-12-31", "2030-01-01"])
    def test_targets_outside_dataset_coverage(self, daily, date):
        with pytest.raises(HistoricalForecastError) as e:
            run_historical_forecast(daily, FAST_CFG, date)
        assert e.value.code == "OUTSIDE_COVERAGE"

    def test_cutoff_before_first_session(self, daily):
        target = daily.index[5]      # cutoff lands a month before the dataset starts
        with pytest.raises(HistoricalForecastError) as e:
            run_historical_forecast(daily, FAST_CFG, target)
        assert e.value.code == "INSUFFICIENT_HISTORY"

    def test_too_little_training_history(self, daily):
        target = daily.index[70]     # ~49 sessions before the cutoff: not enough rows once warm-up and h are removed
        with pytest.raises(HistoricalForecastError) as e:
            run_historical_forecast(daily, FAST_CFG, target)
        assert e.value.code == "INSUFFICIENT_HISTORY"
        assert "training rows" in str(e.value)

    def test_min_train_rows_is_enforced(self, daily):
        with pytest.raises(HistoricalForecastError) as e:
            run_historical_forecast(daily, FAST_CFG, TARGET, min_train_rows=10_000)
        assert e.value.code == "INSUFFICIENT_HISTORY"


# ======================================================================
# 3. Real project data (read-only; skipped if the CSV is not present)
# ======================================================================

CSV_PATH = PROJECT_ROOT / DEFAULT_NIFTYLAB_CONFIG.csv_path


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, parse_dates=["datetime"]).sort_values("datetime").drop_duplicates("datetime")
    return df.set_index("datetime")[["Open", "High", "Low", "Close"]].round(2)


@pytest.mark.skipif(not CSV_PATH.exists(), reason="project NIFTY CSV not present")
class TestOnProjectData:

    def test_no_valid_intraday_target_bridges_a_real_session_break(self, raw):
        idx_s = raw.index.to_series()
        gaps = idx_s.diff()
        breaks = gaps[(gaps > pd.Timedelta(minutes=5)) & (idx_s.dt.date == idx_s.shift().dt.date)]
        assert len(breaks) >= 1, "expected at least one same-date break in the project data"
        for resume_ts in breaks.index:
            pos = raw.index.get_loc(resume_ts)
            last_before = raw.index[pos - 1]
            gap_min = int((resume_ts - last_before) / pd.Timedelta(minutes=1))
            window = raw.iloc[max(0, pos - 80): pos + 80]
            horizons = sorted({5, 15, 30, gap_min})
            out = build_intraday_targets(window, horizons)
            # horizon == gap length: the last bar before the break targets
            # exactly the first bar after it, which EXISTS but is another session.
            assert resume_ts in window.index
            assert pd.isna(out.loc[last_before, f"target_ts_{gap_min}m"])
            for h in horizons:
                crossing = out[(out.index <= last_before) & (out.index + pd.Timedelta(minutes=h) >= resume_ts)]
                assert crossing[f"target_ts_{h}m"].isna().all(), (resume_ts, h)
                assert crossing[f"fwd_ret_{h}m"].isna().all()

    def test_every_valid_intraday_target_is_exact_on_real_bars(self, raw):
        window = raw.iloc[:20_000]
        out = build_intraday_targets(window, [5, 15, 30])
        for h in (5, 15, 30):
            ok = out[f"target_ts_{h}m"].notna()
            tgt = out.loc[ok, f"target_ts_{h}m"]
            assert ((tgt - out.loc[ok].index.to_series()) == pd.Timedelta(minutes=h)).all()
            np.testing.assert_allclose(out.loc[ok, f"target_close_{h}m"].to_numpy(),
                                       window["Close"].loc[tgt].to_numpy())

    def test_historical_forecast_on_real_daily_closes(self, raw):
        daily = pd.DataFrame({
            "Open": raw["Open"].resample("1D").first(), "High": raw["High"].resample("1D").max(),
            "Low": raw["Low"].resample("1D").min(), "Close": raw["Close"].resample("1D").last(),
        }).dropna(how="any")
        target = pd.Timestamp("2025-06-10")
        res = run_historical_forecast(daily, DEFAULT_NIFTYLAB_CONFIG, target)
        assert res.cutoff_date == pd.Timestamp("2025-05-10")
        assert res.last_train_label_date <= res.cutoff_date
        assert res.actual_price == raw.loc[raw.index.normalize() == target, "Close"].iloc[-1]
