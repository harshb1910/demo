"""
Master thesis module:
Evaluating AI-Assisted Demand Forecasting for Inventory Decision-Making in Industrial Supply Chains

This script is fully reproducible and runs end-to-end:
1) Creates (or loads) monthly demand data for two scenarios (normal and volatile)
2) Trains and evaluates ETS + XGBoost forecasting models
3) Simulates inventory decisions using Safety Stock + Reorder Point policy
4) Writes all requested outputs under outputs/
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor


# -----------------------------------------------------------------------------
# Global constants for reproducibility and policy assumptions
# -----------------------------------------------------------------------------
RANDOM_SEED = 42
N_MONTHS = 72
SEASONAL_PERIOD = 12
TEST_HORIZON = 12
LEAD_TIME = 2
Z_VALUE_95 = 1.645  # target cycle service level = 95%


# -----------------------------------------------------------------------------
# Data generation / loading
# -----------------------------------------------------------------------------
def generate_synthetic_monthly_demand(
    n_months: int,
    noise_std: float,
    with_shocks: bool,
    seed: int,
) -> pd.DataFrame:
    """
    Generate synthetic monthly demand with level + trend + seasonality + noise.

    Parameters
    ----------
    n_months : int
        Number of monthly observations.
    noise_std : float
        Standard deviation of Gaussian noise.
    with_shocks : bool
        Whether to inject 2-3 shock months with spikes/drops.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, demand
    """
    rng = np.random.default_rng(seed)

    # Build monthly date index
    dates = pd.date_range(start="2019-01-01", periods=n_months, freq="MS")

    # Signal components
    t = np.arange(n_months)
    level = 220.0
    trend = 1.5 * t
    seasonal_pattern = 35.0 * np.sin(2 * np.pi * t / SEASONAL_PERIOD) + 15.0 * np.cos(
        2 * np.pi * t / SEASONAL_PERIOD
    )
    noise = rng.normal(loc=0.0, scale=noise_std, size=n_months)

    demand = level + trend + seasonal_pattern + noise

    if with_shocks:
        # Inject between 2 and 3 shock months (demand spike or drop)
        n_shocks = int(rng.integers(2, 4))
        shock_indices = rng.choice(np.arange(6, n_months - 6), size=n_shocks, replace=False)
        for idx in shock_indices:
            # Randomly decide spike (+) or drop (-)
            direction = rng.choice([-1, 1])
            magnitude = rng.uniform(0.25, 0.45)  # 25%-45% shock relative to baseline
            demand[idx] = demand[idx] * (1 + direction * magnitude)

    # Enforce non-negative demand
    demand = np.clip(demand, a_min=1.0, a_max=None)

    return pd.DataFrame({"date": dates, "demand": demand})


def load_or_generate_scenarios() -> Dict[str, pd.DataFrame]:
    """
    Load monthly demand CSV if present; otherwise generate synthetic data.

    If local CSV exists at data/monthly_demand.csv, it is used as the base series
    (must include columns: date, demand), then a volatile version is derived by
    adding higher noise + shocks.

    Returns
    -------
    dict
        Keys are scenario names: "normal", "volatile".
        Values are DataFrames with columns: date, demand.
    """
    csv_path = Path("data/monthly_demand.csv")

    if csv_path.exists():
        base = pd.read_csv(csv_path)
        base["date"] = pd.to_datetime(base["date"])
        base = base.sort_values("date").reset_index(drop=True)

        # Ensure exactly 72 months for comparability with thesis design.
        if len(base) >= N_MONTHS:
            base = base.iloc[-N_MONTHS:].copy()
        else:
            # If fewer than 72 months, regenerate to respect requirement scope.
            base = generate_synthetic_monthly_demand(
                n_months=N_MONTHS,
                noise_std=8.0,
                with_shocks=False,
                seed=RANDOM_SEED,
            )

        normal_df = base[["date", "demand"]].copy()

        rng = np.random.default_rng(RANDOM_SEED + 100)
        volatile_demand = normal_df["demand"].values + rng.normal(0.0, 18.0, len(normal_df))

        # Inject 2-3 shocks in volatile scenario
        n_shocks = int(rng.integers(2, 4))
        shock_idx = rng.choice(np.arange(6, len(normal_df) - 6), size=n_shocks, replace=False)
        for idx in shock_idx:
            direction = rng.choice([-1, 1])
            magnitude = rng.uniform(0.25, 0.45)
            volatile_demand[idx] *= 1 + direction * magnitude

        volatile_df = pd.DataFrame(
            {
                "date": normal_df["date"],
                "demand": np.clip(volatile_demand, 1.0, None),
            }
        )

    else:
        # Required synthetic scenarios
        normal_df = generate_synthetic_monthly_demand(
            n_months=N_MONTHS,
            noise_std=8.0,
            with_shocks=False,
            seed=RANDOM_SEED,
        )
        volatile_df = generate_synthetic_monthly_demand(
            n_months=N_MONTHS,
            noise_std=20.0,
            with_shocks=True,
            seed=RANDOM_SEED + 1,
        )

    return {"normal": normal_df, "volatile": volatile_df}


# -----------------------------------------------------------------------------
# Forecasting utilities (Section 4.1)
# -----------------------------------------------------------------------------
def train_test_split_time(df: pd.DataFrame, test_horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based split with last N months as test."""
    df = df.sort_values("date").reset_index(drop=True)
    train = df.iloc[:-test_horizon].copy()
    test = df.iloc[-test_horizon:].copy()
    return train, test


def build_xgb_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag, rolling, and calendar features for XGBoost.

    Rolling features are derived from demand.shift(1) to avoid leakage.
    """
    feat = df.copy()

    # Lag features
    for lag in [1, 2, 3, 6, 12]:
        feat[f"lag_{lag}"] = feat["demand"].shift(lag)

    # Shift once before rolling windows to avoid using current-month demand
    shifted = feat["demand"].shift(1)
    for w in [3, 6, 12]:
        feat[f"roll_mean_{w}"] = shifted.rolling(window=w).mean()
        feat[f"roll_std_{w}"] = shifted.rolling(window=w).std()

    # Calendar features
    feat["month"] = feat["date"].dt.month
    feat["quarter"] = feat["date"].dt.quarter
    feat["year"] = feat["date"].dt.year

    return feat


def fit_forecasting_models(df: pd.DataFrame) -> Dict[str, object]:
    """
    Fit ETS and XGBoost models and evaluate on test set.

    Returns dictionary containing metrics, forecasts, train residual sigma,
    and data needed for downstream inventory simulation.
    """
    train_df, test_df = train_test_split_time(df, TEST_HORIZON)

    # ------------------------
    # Model 1: ETS (add-add)
    # ------------------------
    ets_model = ExponentialSmoothing(
        train_df["demand"],
        trend="add",
        seasonal="add",
        seasonal_periods=SEASONAL_PERIOD,
    ).fit(optimized=True)

    ets_test_forecast = ets_model.forecast(TEST_HORIZON)
    ets_train_fitted = ets_model.fittedvalues
    ets_train_residuals = train_df["demand"].values - ets_train_fitted
    ets_sigma = float(np.std(ets_train_residuals, ddof=1))

    # -----------------------------------------
    # Model 2: XGBoost with engineered features
    # -----------------------------------------
    feat_df = build_xgb_features(df)

    feature_cols = [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_6",
        "lag_12",
        "roll_mean_3",
        "roll_mean_6",
        "roll_mean_12",
        "roll_std_3",
        "roll_std_6",
        "roll_std_12",
        "month",
        "quarter",
        "year",
    ]

    # Use full historical frame with NaN rows dropped for valid feature rows.
    valid = feat_df.dropna().reset_index(drop=True)

    # Train/test split by date so that last 12 calendar months remain the test horizon.
    test_start_date = test_df["date"].min()
    xgb_train = valid[valid["date"] < test_start_date].copy()
    xgb_test = valid[valid["date"] >= test_start_date].copy()

    xgb_model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
    )
    xgb_model.fit(xgb_train[feature_cols], xgb_train["demand"])

    xgb_train_pred = xgb_model.predict(xgb_train[feature_cols])
    xgb_train_residuals = xgb_train["demand"].values - xgb_train_pred
    xgb_sigma = float(np.std(xgb_train_residuals, ddof=1))

    xgb_test_forecast = xgb_model.predict(xgb_test[feature_cols])

    # Align test series and forecasts to a single table
    merged_test = test_df[["date", "demand"]].rename(columns={"demand": "actual"}).copy()
    merged_test["forecast_ETS"] = ets_test_forecast.values

    # XGB test rows should match the same horizon; align by date for safety.
    xgb_map = dict(zip(xgb_test["date"], xgb_test_forecast))
    merged_test["forecast_XGBoost"] = merged_test["date"].map(xgb_map)

    # Rare edge fallback in case a date is missing due to features warm-up.
    merged_test["forecast_XGBoost"] = merged_test["forecast_XGBoost"].ffill().bfill()

    # Metrics
    mae_ets = mean_absolute_error(merged_test["actual"], merged_test["forecast_ETS"])
    rmse_ets = np.sqrt(mean_squared_error(merged_test["actual"], merged_test["forecast_ETS"]))

    mae_xgb = mean_absolute_error(merged_test["actual"], merged_test["forecast_XGBoost"])
    rmse_xgb = np.sqrt(mean_squared_error(merged_test["actual"], merged_test["forecast_XGBoost"]))

    metrics_df = pd.DataFrame(
        {
            "model": ["ETS", "XGBoost"],
            "MAE": [mae_ets, mae_xgb],
            "RMSE": [rmse_ets, rmse_xgb],
            "train_residual_sigma": [ets_sigma, xgb_sigma],
        }
    )

    return {
        "train_df": train_df,
        "test_df": test_df,
        "forecasts_test": merged_test,
        "metrics": metrics_df,
        "sigma_by_model": {"ETS": ets_sigma, "XGBoost": xgb_sigma},
    }


def plot_forecasts(forecasts_test: pd.DataFrame, scenario: str, out_dir: Path) -> None:
    """Save forecast comparison plot for a scenario."""
    plt.figure(figsize=(11, 5))
    plt.plot(forecasts_test["date"], forecasts_test["actual"], marker="o", label="Actual")
    plt.plot(forecasts_test["date"], forecasts_test["forecast_ETS"], marker="s", label="ETS")
    plt.plot(
        forecasts_test["date"],
        forecasts_test["forecast_XGBoost"],
        marker="^",
        label="XGBoost",
    )
    plt.title(f"Test Period Forecasts ({scenario.capitalize()} scenario)")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"forecast_plot_{scenario}.png", dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# Inventory decision simulation (Section 4.2)
# -----------------------------------------------------------------------------
def repeated_forecast_value(forecast_arr: np.ndarray, idx: int) -> float:
    """Return forecast at idx; if out-of-range, repeat last available forecast."""
    if idx < len(forecast_arr):
        return float(forecast_arr[idx])
    return float(forecast_arr[-1])


def simulate_inventory(
    dates: np.ndarray,
    actual_demand: np.ndarray,
    forecast: np.ndarray,
    sigma: float,
    model_name: str,
    lead_time: int = LEAD_TIME,
    z_value: float = Z_VALUE_95,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Simulate monthly inventory with continuous review style decision logic.

    Rules applied per month t:
    - SafetyStock = z * sigma * sqrt(LT)
    - Expected lead-time demand = sum(forecast[t : t+LT], repeat last forecast if needed)
    - ROP = ExpectedLeadTimeDemand + SafetyStock
    - Target level S = ExpectedLeadTimeDemand + forecast_next_1_month + SafetyStock
    - Place order if inventory position <= ROP, ordering up to S.

    Returns
    -------
    sim_df : pd.DataFrame
        Detailed month-level inventory simulation output.
    kpis : dict
        KPI summary for the given model.
    """
    n = len(actual_demand)
    safety_stock = float(z_value * sigma * np.sqrt(lead_time))

    # Precompute policy thresholds per month from forecast trajectory
    rop_vals = []
    s_vals = []
    for t in range(n):
        expected_lt_demand = sum(repeated_forecast_value(forecast, t + k) for k in range(lead_time))
        next_1 = repeated_forecast_value(forecast, t)
        rop = expected_lt_demand + safety_stock
        s_level = expected_lt_demand + next_1 + safety_stock
        rop_vals.append(rop)
        s_vals.append(s_level)

    # Inventory state
    on_hand = float(s_vals[0])  # initial on-hand = S at first test month
    open_orders: Dict[int, float] = {}  # key: arrival month index, value: quantity

    records = []

    total_stockout_units = 0.0
    stockout_months = 0
    stock_up_months = 0
    on_hand_history = []

    for t in range(n):
        # Receive arriving order(s)
        arriving_qty = open_orders.pop(t, 0.0)
        on_hand += arriving_qty

        # Observe demand and consume inventory
        dem = float(actual_demand[t])
        if dem > on_hand:
            stockout_units = dem - on_hand
            on_hand = 0.0
            stockout_months += 1
        else:
            stockout_units = 0.0
            on_hand -= dem

        total_stockout_units += stockout_units

        # On-order after any receipt, before placing new order
        on_order_before = float(sum(open_orders.values()))

        # Inventory position: on-hand + pipeline (lost-sales setting, no backlog carry)
        inventory_position = on_hand + on_order_before

        rop_t = float(rop_vals[t])
        s_t = float(s_vals[t])

        # Continuous review reorder trigger
        if inventory_position <= rop_t:
            order_qty = max(0.0, s_t - inventory_position)
            arrival_idx = t + lead_time
            open_orders[arrival_idx] = open_orders.get(arrival_idx, 0.0) + order_qty
        else:
            order_qty = 0.0

        on_order_after = float(sum(open_orders.values()))

        if on_hand > s_t:
            stock_up_months += 1

        on_hand_history.append(on_hand)

        records.append(
            {
                "model": model_name,
                "date": dates[t],
                "demand": dem,
                "on_hand": on_hand,
                "on_order": on_order_after,
                "ROP": rop_t,
                "S": s_t,
                "order_qty": order_qty,
                "stockout_units": stockout_units,
            }
        )

    total_demand = float(np.sum(actual_demand))
    fill_rate = 1.0 - (total_stockout_units / total_demand if total_demand > 0 else 0.0)
    cycle_service = 1.0 - (stockout_months / n if n > 0 else 0.0)
    avg_on_hand = float(np.mean(on_hand_history)) if on_hand_history else 0.0

    kpis = {
        "model": model_name,
        "fill_rate": fill_rate,
        "cycle_service_level": cycle_service,
        "stockout_months": stockout_months,
        "stock_up_months": stock_up_months,
        "average_on_hand_inventory": avg_on_hand,
    }

    return pd.DataFrame(records), kpis


def run_inventory_module(
    forecasts_test: pd.DataFrame,
    sigma_by_model: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run simulation for ETS and XGBoost, returning simulation rows and KPI table."""
    dates = forecasts_test["date"].values
    actual = forecasts_test["actual"].values.astype(float)

    simulation_frames = []
    kpi_rows = []

    for model, forecast_col in [("ETS", "forecast_ETS"), ("XGBoost", "forecast_XGBoost")]:
        sim_df, kpis = simulate_inventory(
            dates=dates,
            actual_demand=actual,
            forecast=forecasts_test[forecast_col].values.astype(float),
            sigma=float(sigma_by_model[model]),
            model_name=model,
            lead_time=LEAD_TIME,
            z_value=Z_VALUE_95,
        )
        simulation_frames.append(sim_df)
        kpi_rows.append(kpis)

    simulation_all = pd.concat(simulation_frames, ignore_index=True)
    kpi_df = pd.DataFrame(kpi_rows)
    return simulation_all, kpi_df


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def run_pipeline() -> None:
    """Execute all thesis sections and write outputs for each scenario."""
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = load_or_generate_scenarios()

    for scenario_name, df in scenarios.items():
        # Ensure expected shape and column order
        scenario_df = df[["date", "demand"]].copy().sort_values("date").reset_index(drop=True)

        # Forecasting section (4.1)
        forecast_results = fit_forecasting_models(scenario_df)
        metrics_df = forecast_results["metrics"]
        forecasts_test_df = forecast_results["forecasts_test"]

        metrics_df.to_csv(out_dir / f"forecast_metrics_{scenario_name}.csv", index=False)
        forecasts_test_df.to_csv(
            out_dir / f"forecasts_test_period_{scenario_name}.csv", index=False
        )
        plot_forecasts(forecasts_test_df, scenario_name, out_dir)

        # Inventory section (4.2)
        sim_df, kpi_df = run_inventory_module(
            forecasts_test=forecasts_test_df,
            sigma_by_model=forecast_results["sigma_by_model"],
        )

        kpi_df.to_csv(out_dir / f"inventory_kpis_{scenario_name}.csv", index=False)
        sim_df.to_csv(out_dir / f"inventory_simulation_{scenario_name}.csv", index=False)

        print(f"Completed scenario: {scenario_name}")


if __name__ == "__main__":
    run_pipeline()
