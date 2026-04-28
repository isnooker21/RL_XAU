"""
v12_monte_carlo.py
==================
Vectorized Monte Carlo path engine for V12 Genesis.

`run_mc_summary` simulates many GBM paths from recent log-returns and returns:
  - confidence: P(terminal log-return > 0)  [0,1]
  - margin_at_risk: tail severity of minimum log-path vs start (5th percentile)
  - path_var_95: 95% VaR on simple returns at horizon (positive number = loss magnitude)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from v12_config import V12Config, CONFIG


def _estimate_mu_sigma(log_close: np.ndarray) -> tuple[float, float]:
    log_close = np.asarray(log_close, dtype=np.float64).ravel()
    if log_close.size < 4:
        return 0.0, 1e-4
    r = np.diff(np.log(np.clip(log_close, 1e-12, None)))
    if r.size < 2:
        return 0.0, 1e-4
    mu = float(np.mean(r))
    sig = float(np.std(r))
    sig = max(sig, 1e-6)
    return mu, sig


class MonteCarloEngine:
    """Stateful wrapper; core math is in `simulate_paths` / `run_mc_summary`."""

    def __init__(self, cfg: Optional[V12Config] = None):
        self.cfg = cfg or CONFIG

    def run(
        self,
        close_path: np.ndarray,
        rng: np.random.Generator,
        *,
        n_paths: Optional[int] = None,
    ) -> Dict[str, float]:
        return run_mc_summary(close_path, self.cfg, rng, n_paths=n_paths)


def simulate_paths(
    close_path: np.ndarray,
    cfg: V12Config,
    rng: np.random.Generator,
    *,
    n_paths: Optional[int] = None,
    stress: bool = False,
) -> Dict[str, Any]:
    """
    Simulate correlated-ish GBM steps (iid shocks) for n_paths × H.

    Returns dict with:
      terminal_log_ret: (n_paths,) log(S_H/S_0) with S_0 = last close
      min_log_cum: (n_paths,) min cumulative log increment along path
      per_step_shocks: (n_paths, H) optional for debugging
    """
    close_path = np.asarray(close_path, dtype=np.float64).ravel()
    n_paths = int(cfg.monte_carlo.n_paths if n_paths is None else n_paths)
    H = int(cfg.monte_carlo.horizon_steps)
    if close_path.size < 8 or H < 1 or n_paths < 1:
        return dict(
            terminal_log_ret=np.zeros(0),
            min_log_cum=np.zeros(0),
        )

    mu, sig = _estimate_mu_sigma(close_path)
    dt = float(cfg.monte_carlo.dt_fraction_of_bar)

    if stress:
        lo, hi = cfg.monte_carlo.stress_spread_mult_range
        sig *= float(rng.uniform(lo, hi))
        sig *= float(cfg.monte_carlo.stress_jump_sigma_mult ** 0.25)

    # GBM on log increment per step: r_t ~ N((mu - 0.5*sig^2)*dt, sig^2*dt)
    drift = (mu - 0.5 * sig * sig) * dt
    vol = sig * np.sqrt(dt)
    z = rng.standard_normal((n_paths, H))
    inc = drift + vol * z
    log_cum = np.cumsum(inc, axis=1)
    terminal_log_ret = log_cum[:, -1]
    min_log_cum = np.min(log_cum, axis=1)
    return dict(
        terminal_log_ret=terminal_log_ret.astype(np.float64),
        min_log_cum=min_log_cum.astype(np.float64),
        inc=inc,
    )


def run_mc_summary(
    close_path: np.ndarray,
    cfg: V12Config,
    rng: np.random.Generator,
    *,
    n_paths: Optional[int] = None,
) -> Dict[str, float]:
    """
    Public entry used by `v12_physics_env.V12PhysicsEnv._monte_carlo_summary`.

    confidence: fraction of paths with positive terminal log return vs now
    margin_at_risk: -quantile_5(min_log_cum)  (larger => worse tail drawdowns)
    path_var_95: -quantile_5(simple return) where simple_ret = exp(terminal_log_ret)-1
    """
    out = simulate_paths(close_path, cfg, rng, n_paths=n_paths, stress=False)
    term = out.get("terminal_log_ret", np.zeros(0))
    mini = out.get("min_log_cum", np.zeros(0))
    if term.size == 0:
        return {"confidence": 0.5, "margin_at_risk": 0.0, "path_var_95": 0.0}

    confidence = float(np.mean(term > 0.0))
    # Tail "bad" min cumulative move (more negative = worse)
    q5_min = float(np.percentile(mini, 5))
    margin_at_risk = float(max(0.0, -q5_min))

    simple = np.expm1(term)
    q5_ret = float(np.percentile(simple, 5))
    path_var_95 = float(max(0.0, -q5_ret))

    return {
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "margin_at_risk": margin_at_risk,
        "path_var_95": path_var_95,
    }


__all__ = [
    "MonteCarloEngine",
    "simulate_paths",
    "run_mc_summary",
]
