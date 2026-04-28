"""
v12_physics_env.py
==================
V12 Genesis — Gymnasium environment: physics, Hurst, Z-score masks, Monte Carlo,
simulated execution with **zero-cutting** agent actions (Stay | Buy | Sell).

System layer (not RL actions): delta-neutral hedge, Houdini profit recycling,
adversarial shocks during training.

Hard rule: margin level < 100% → Buy/Sell masked.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from v12_config import V12Config, CONFIG


class V12Action(IntEnum):
    STAY = 0
    BUY = 1
    SELL = 2


@dataclass
class SimPosition:
    ticket: int
    side: int  # +1 long, -1 short
    lot: float
    entry_price: float
    open_step: int


class V12PhysicsEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    REQUIRED_COLS: Tuple[str, ...] = ("open", "high", "low", "close", "volume")

    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[V12Config] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        random_start: bool = True,
        mc_n_paths: Optional[int] = None,
    ):
        super().__init__()
        self.cfg = config or CONFIG
        self.render_mode = render_mode
        self._rng = np.random.default_rng(seed)
        self.random_start = bool(random_start)
        self._mc_n_paths = mc_n_paths if mc_n_paths is not None else self.cfg.mc_n_paths_train

        self._validate_df(df)
        self.df = self._normalize_datetime(df)
        self.n_bars = len(self.df)

        self._o = self.df["open"].to_numpy(np.float64)
        self._h = self.df["high"].to_numpy(np.float64)
        self._l = self.df["low"].to_numpy(np.float64)
        self._c = self.df["close"].to_numpy(np.float64)
        self._v = self.df["volume"].to_numpy(np.float64)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.cfg.n_obs_features,),
            dtype=np.float32,
        )

        self._t = 0
        self._episode_start = 0
        self._valid_mask = np.ones(3, dtype=bool)
        self._margin_level = 10_000.0
        self._equity = float(self.cfg.initial_capital)
        self._balance = float(self.cfg.initial_capital)
        self._peak_equity = float(self.cfg.initial_capital)
        self._positions: List[SimPosition] = []
        self._next_ticket = 1
        self._episode_opens = 0
        self._bars_in_episode = 0
        self._realized_step = 0.0
        self._recycle_pool = 0.0
        self._spread_mult = 1.0
        self._last_chaos = False
        self._last_mc: Dict[str, float] = {}

    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        opts = options or {}
        if opts.get("start_step") is not None:
            start = int(opts["start_step"])
        elif self.random_start:
            hi = max(self.cfg.lookback_bars + 10, self.n_bars - 500)
            start = int(self._rng.integers(self.cfg.lookback_bars, hi))
        else:
            start = self.cfg.lookback_bars
        start = max(start, self.cfg.lookback_bars)
        start = min(start, self.n_bars - 5)

        self._episode_start = start
        self._t = start
        self._balance = float(self.cfg.initial_capital)
        self._equity = self._balance
        self._peak_equity = self._balance
        self._positions.clear()
        self._next_ticket = 1
        self._episode_opens = 0
        self._bars_in_episode = 0
        self._realized_step = 0.0
        self._recycle_pool = 0.0
        self._spread_mult = 1.0
        self._last_chaos = False
        self._last_mc = {}
        self._mark_mid = None
        self._sync_account()
        obs = self._build_observation(self._t)
        self._valid_mask = self._compute_action_masks(self._t)
        info = self._info_dict()
        info["action_mask"] = self._valid_mask.copy()
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if not self._valid_mask[action]:
            action = int(V12Action.STAY)

        self._spread_mult = 1.0
        self._apply_physics_and_chaos()
        self._realized_step = 0.0
        z_action = self._zscore(self._t)

        self._agent_open(action)
        self._system_delta_hedge()
        self._system_houdini_recycle()

        self._advance_mark_mid()
        self._t += 1
        self._bars_in_episode += 1

        terminated = False
        truncated = self._t >= self.n_bars - 1
        if self.cfg.episode_max_steps is not None:
            truncated = truncated or (self._bars_in_episode >= self.cfg.episode_max_steps)

        self._sync_account()
        self._system_emergency_deleverage()
        reward = self._compute_reward(action, z_action)

        obs = self._build_observation(self._t)
        self._valid_mask = self._compute_action_masks(self._t)
        info = self._info_dict()
        info["action_mask"] = self._valid_mask.copy()
        self._last_chaos = False
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self._valid_mask.copy()

    def render(self) -> None:
        if self.render_mode == "human":
            print(
                f"t={self._t} margin={self._margin_level:.2f}% "
                f"eq={self._equity:.2f} n_pos={len(self._positions)} mask={self._valid_mask}"
            )

    # ------------------------------------------------------------------
    def _full_spread_price(self) -> float:
        return (
            self.cfg.broker.spread_points
            * self.cfg.broker.point_size
            * self._spread_mult
        )

    def _half_spread(self) -> float:
        return 0.5 * self._full_spread_price()

    def _contract_value_per_lot(self, price: float) -> float:
        return price * self.cfg.broker.contract_size

    def _used_margin(self, price: float) -> float:
        lev = max(self.cfg.broker.leverage, 1e-9)
        um = 0.0
        for p in self._positions:
            um += abs(p.lot) * self._contract_value_per_lot(price) / lev
        return um

    def _floating_pnl_mid(self, mid: float) -> float:
        pt = self.cfg.broker.contract_size
        s = 0.0
        for p in self._positions:
            s += p.side * (mid - p.entry_price) * p.lot * pt
        return s

    def _sync_account(self) -> None:
        mark = getattr(self, "_mark_mid", None)
        if mark is not None:
            mid = float(mark)
            self._mark_mid = None
        else:
            mid = float(self._c[min(self._t, self.n_bars - 1)])
        self._floating_pnl = self._floating_pnl_mid(mid)
        self._equity = self._balance + self._floating_pnl
        self._peak_equity = max(self._peak_equity, self._equity)
        um = self._used_margin(mid)
        if um <= 1e-12:
            self._margin_level = 10_000.0
        else:
            self._margin_level = 100.0 * self._equity / um

    def _advance_mark_mid(self) -> None:
        """Synthetic mark price after bar t→t+1 including adversarial return shock."""
        self._mark_mid = None
        if self._t + 1 >= self.n_bars:
            self._mark_mid = float(self._c[self._t])
            return
        c0 = float(self._c[self._t])
        c1 = float(self._c[self._t + 1])
        r = (c1 - c0) / max(abs(c0), 1e-12)
        r += float(getattr(self, "_chaos_ret_shock", 0.0))
        self._chaos_ret_shock = 0.0
        self._mark_mid = float(c0 * (1.0 + r))

    def _apply_physics_and_chaos(self) -> None:
        self._chaos_ret_shock = 0.0
        self._last_chaos = False
        if not self.cfg.adversarial.enabled:
            return
        if self._rng.random() > self.cfg.adversarial.chaos_prob_per_step:
            return
        self._last_chaos = True
        lo, hi = self.cfg.adversarial.spread_widen_mult_range
        self._spread_mult = float(self._rng.uniform(lo, hi))
        smin, smax = self.cfg.adversarial.spike_sigma_mult_range
        mult = float(self._rng.uniform(smin, smax))
        # shock on log return ~ N(0, (mult*sigma_bar)^2) estimated from last 16 bars
        start = max(0, self._t - 16)
        seg = self._c[start : self._t + 1]
        if seg.size >= 4:
            lr = np.diff(np.log(np.clip(seg, 1e-12, None)))
            sig = float(np.std(lr)) if lr.size else 0.001
        else:
            sig = 0.001
        self._chaos_ret_shock = float(self._rng.normal(0.0, mult * sig))

    def _agent_open(self, action: int) -> None:
        if action == int(V12Action.STAY):
            return
        if not self._valid_mask[action]:
            return
        if len(self._positions) >= self.cfg.sim.max_open_positions:
            return
        mid = float(self._c[self._t])
        lot = float(self.cfg.sim.min_lot)
        if action == int(V12Action.BUY) and not self._can_add_lot(side=1, lot=lot):
            return
        if action == int(V12Action.SELL) and not self._can_add_lot(side=-1, lot=lot):
            return
        hs = self._half_spread()
        comm = self.cfg.broker.commission_per_lot * lot

        if action == int(V12Action.BUY):
            entry = mid + hs
            self._positions.append(
                SimPosition(
                    ticket=self._next_ticket,
                    side=1,
                    lot=lot,
                    entry_price=entry,
                    open_step=self._t,
                )
            )
        elif action == int(V12Action.SELL):
            entry = mid - hs
            self._positions.append(
                SimPosition(
                    ticket=self._next_ticket,
                    side=-1,
                    lot=lot,
                    entry_price=entry,
                    open_step=self._t,
                )
            )
        else:
            return
        self._next_ticket += 1
        self._balance -= comm
        self._episode_opens += 1
        self._sync_account()

    def _net_lots(self) -> float:
        return float(sum(p.side * p.lot for p in self._positions))

    def _can_add_lot(self, side: int, lot: float) -> bool:
        cap = max(float(self.cfg.sim.max_abs_net_lots), 0.0)
        if cap <= 0:
            return True
        projected = self._net_lots() + float(side) * float(lot)
        return abs(projected) <= cap + 1e-9

    def _system_delta_hedge(self) -> None:
        """Minimum-style hedge opposite net when margin weak and MC tail is bad."""
        mc = self._monte_carlo_summary(self._t)
        self._last_mc = mc
        mar = mc.get("margin_at_risk", 0.0)
        if self._margin_level > self.cfg.risk.mc_hedge_margin_level_max:
            return
        if mar < self.cfg.risk.mc_hedge_mar_threshold:
            return
        net = self._net_lots()
        if abs(net) < self.cfg.sim.min_lot * 0.5:
            return
        mid = float(self._c[self._t])
        lot = float(self.cfg.sim.min_lot)
        hs = self._half_spread()
        comm = self.cfg.broker.commission_per_lot * lot
        if net > 0:
            entry = mid - hs
            side = -1
        else:
            entry = mid + hs
            side = 1
        if not self._can_add_lot(side=side, lot=lot):
            return
        self._positions.append(
            SimPosition(
                ticket=self._next_ticket,
                side=side,
                lot=lot,
                entry_price=entry,
                open_step=self._t,
            )
        )
        self._next_ticket += 1
        self._balance -= comm
        self._sync_account()

    def _close_worst_loser_chunk(self, lot_chunk: float) -> bool:
        if not self._positions:
            return False
        mid = float(self._c[self._t])
        pt = self.cfg.broker.contract_size
        worst_idx = -1
        worst_pnl = 0.0
        for i, p in enumerate(self._positions):
            pnl = p.side * (mid - p.entry_price) * p.lot * pt
            if i == 0 or pnl < worst_pnl:
                worst_pnl = pnl
                worst_idx = i
        if worst_idx < 0:
            return False
        pos = self._positions[worst_idx]
        close_lot = min(float(lot_chunk), pos.lot)
        if close_lot <= 0:
            return False
        realized = pos.side * (mid - pos.entry_price) * close_lot * pt
        realized -= self.cfg.broker.commission_per_lot * close_lot
        self._balance += realized
        self._realized_step += realized
        pos.lot -= close_lot
        if pos.lot <= 1e-9:
            self._positions.pop(worst_idx)
        return True

    def _system_emergency_deleverage(self) -> None:
        """
        Harder survival layer:
        if margin drops below threshold, force-trim worst losers to cut used margin.
        """
        trigger = float(self.cfg.sim.emergency_deleverage_margin_level)
        if self._margin_level >= trigger:
            return
        max_loops = 8
        lot_chunk = float(self.cfg.sim.emergency_close_lot)
        for _ in range(max_loops):
            if self._margin_level >= trigger or not self._positions:
                break
            changed = self._close_worst_loser_chunk(lot_chunk)
            if not changed:
                break
            self._sync_account()

    def _system_houdini_recycle(self) -> None:
        """Use recycle pool / session edge to chip worst underwater leg."""
        if not self._positions:
            return
        mid = float(self._c[self._t])
        pt = self.cfg.broker.contract_size
        pnls: List[Tuple[float, int]] = []
        for i, p in enumerate(self._positions):
            fl = p.side * (mid - p.entry_price) * p.lot * pt
            pnls.append((fl, i))
        pnls.sort(key=lambda x: x[0])
        worst_idx = pnls[0][1]
        worst_p = self._positions[worst_idx]
        worst_fl = pnls[0][0]
        if worst_fl >= -1.0:
            return
        if self._balance < self.cfg.initial_capital + self.cfg.sim.houdini_buffer_min:
            return
        lot_close = min(self.cfg.sim.houdini_close_lot, worst_p.lot)
        if lot_close <= 0:
            return
        exit_px = mid
        realized = worst_p.side * (exit_px - worst_p.entry_price) * lot_close * pt
        realized -= self.cfg.broker.commission_per_lot * lot_close
        self._balance += realized
        self._realized_step += realized
        if realized > 0:
            self._recycle_pool += realized * 0.25
        worst_p.lot -= lot_close
        if worst_p.lot <= 1e-9:
            self._positions.pop(worst_idx)
        self._sync_account()

    def _compute_velocity_acceleration_energy(
        self, t: int
    ) -> Tuple[float, float, float]:
        w = self.cfg.lookback_bars
        start = max(0, t - w + 1)
        px = self._c[start : t + 1].astype(np.float64)
        if self.cfg.physics.use_log_price:
            px = np.log(np.clip(px, 1e-12, None))
        if len(px) < max(self.cfg.physics.velocity_window + 2, 8):
            return 0.0, 0.0, 0.0
        vw = self.cfg.physics.velocity_window
        v = float((px[-1] - px[-1 - vw]) / max(vw, 1))
        v_prev = float((px[-2] - px[-2 - vw]) / max(vw, 1))
        a = v - v_prev
        m = self.cfg.physics.kinetic_mass
        ek = 0.5 * m * v * v
        return v, a, ek

    def _rolling_hurst(self, t: int) -> float:
        w = min(self.cfg.hurst.window, t + 1)
        w = max(w, self.cfg.hurst.min_window)
        start = t - w + 1
        logp = np.log(np.clip(self._c[start : t + 1], 1e-12, None))
        r = np.diff(logp)
        if r.size < 16:
            return 0.5
        lag = max(2, w // 32)
        x0, x1 = r[:-lag], r[lag:]
        c = np.corrcoef(x0, x1)[0, 1]
        c = float(np.clip(c, -0.99, 0.99))
        h = 0.5 + 0.25 * np.sign(c) * (abs(c) ** 0.5)
        return float(np.clip(h, 0.0, 1.0))

    def _zscore(self, t: int) -> float:
        lb = self.cfg.zscore.lookback
        start = max(0, t - lb)
        window = self._c[start:t].astype(np.float64)
        if window.size < 8:
            return 0.0
        mu = float(window.mean())
        sig = float(window.std()) or 1e-9
        z = (float(self._c[t]) - mu) / sig
        return float(np.clip(z, -self.cfg.zscore.clip_sigma, self.cfg.zscore.clip_sigma))

    def _monte_carlo_summary(self, t: int) -> Dict[str, float]:
        from v12_monte_carlo import run_mc_summary

        path = self._c[max(0, t - 256) : t + 1]
        return run_mc_summary(path, self.cfg, self._rng, n_paths=self._mc_n_paths)

    def _dynamic_grid_distance_pips(self, t: int) -> float:
        _, a, _ = self._compute_velocity_acceleration_energy(t)
        lb = self.cfg.grid.vol_lookback
        start = max(0, t - lb)
        seg = self._c[start : t + 1]
        if seg.size < 4:
            vol = 0.0
        else:
            lr = np.diff(np.log(np.clip(seg, 1e-12, None)))
            vol = float(lr.std()) if lr.size else 0.0
        base = self.cfg.grid.base_distance_pips
        scale = 1.0 + 8.0 * vol + 0.5 * abs(a) * 1e4
        d = base * scale
        return float(
            np.clip(
                d,
                self.cfg.grid.min_distance_pips,
                self.cfg.grid.max_distance_pips,
            )
        )

    def _compute_action_masks(self, t: int) -> np.ndarray:
        mask = np.ones(3, dtype=bool)
        z = self._zscore(t)
        if z >= self.cfg.zscore.buy_mask_z:
            mask[int(V12Action.BUY)] = False
        if z <= self.cfg.zscore.sell_mask_z:
            mask[int(V12Action.SELL)] = False
        if self._margin_level < self.cfg.risk.margin_level_min_open:
            mask[int(V12Action.BUY)] = False
            mask[int(V12Action.SELL)] = False
        if not mask.any():
            mask[int(V12Action.STAY)] = True
        return mask

    def _compute_reward(self, action: int, z_action: float) -> float:
        rspec = self.cfg.reward
        cap = max(self.cfg.initial_capital, 1.0)
        rew = rspec.scale_realized_pnl * (self._realized_step / cap)

        ml = self._margin_level
        if ml >= self.cfg.risk.margin_level_min_open:
            rew += rspec.margin_bonus_per_pct_above_guard * (ml - self.cfg.risk.margin_level_min_open) / 100.0
        else:
            rew -= rspec.margin_penalty_below_guard

        dd = 1.0 - (self._equity / max(self._peak_equity, 1e-9))
        dd = max(0.0, float(dd))
        if dd > 0:
            rew -= rspec.drawdown_penalty_weight * (dd / max(self.cfg.risk.target_max_drawdown_pct, 1e-6))

        if action in (int(V12Action.BUY), int(V12Action.SELL)):
            rew -= rspec.trade_open_penalty

        if action == int(V12Action.BUY) and z_action >= self.cfg.zscore.buy_mask_z:
            rew -= rspec.doi_penalty_weight
        if action == int(V12Action.SELL) and z_action <= self.cfg.zscore.sell_mask_z:
            rew -= rspec.doi_penalty_weight

        if self._last_chaos and ml >= self.cfg.risk.margin_level_min_open:
            rew += rspec.chaos_survive_bonus

        days = max(self._bars_in_episode / max(self.cfg.sim.bars_per_day, 1e-6), 1e-6)
        tpd = self._episode_opens / days
        mid = self.cfg.risk.target_trades_per_day_min + self.cfg.risk.target_trades_per_day_max
        mid *= 0.5
        err = abs(tpd - rspec.tpd_target_mid) / max(mid, 1e-6)
        err = min(err, float(rspec.tpd_shaping_max_err))
        rew -= rspec.tpd_shaping_weight * float(err)

        return float(rew)

    def _build_observation(self, t: int) -> np.ndarray:
        cap = max(abs(self.cfg.initial_capital), 1.0)
        v, a, ek = self._compute_velocity_acceleration_energy(t)
        h = self._rolling_hurst(t)
        z = self._zscore(t)
        mc = self._last_mc if self._last_mc else self._monte_carlo_summary(t)
        grid_d = self._dynamic_grid_distance_pips(t)
        net = self._net_lots()
        npos = len(self._positions)
        days = max(self._bars_in_episode / max(self.cfg.sim.bars_per_day, 1e-6), 1e-6)
        tpd = self._episode_opens / days

        feat = np.array(
            [
                v,
                a,
                ek,
                h,
                z,
                mc.get("confidence", 0.5),
                mc.get("margin_at_risk", 0.0),
                mc.get("path_var_95", 0.0),
                grid_d / 100.0,
                self._margin_level / 1000.0,
                self._equity / self.cfg.initial_capital,
                net,
                float(t) / max(self.n_bars, 1),
                float(npos) / max(self.cfg.sim.max_open_positions, 1),
                float(tpd) / 5.0,
                float(self._floating_pnl) / cap,
                1.0 if h > self.cfg.hurst.trending_threshold else 0.0,
                1.0 if h < self.cfg.hurst.mean_revert_threshold else 0.0,
            ],
            dtype=np.float32,
        )
        out = np.zeros(self.cfg.n_obs_features, dtype=np.float32)
        n = min(len(feat), self.cfg.n_obs_features)
        out[:n] = feat[:n]
        return out

    def _info_dict(self) -> Dict[str, Any]:
        return {
            "margin_level": self._margin_level,
            "equity": self._equity,
            "balance": self._balance,
            "net_lots": self._net_lots(),
            "n_positions": len(self._positions),
            "action_mask": self._valid_mask.copy(),
            "mc": dict(self._last_mc),
        }

    @staticmethod
    def _validate_df(df: pd.DataFrame) -> None:
        missing = [c for c in V12PhysicsEnv.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"df missing columns: {missing}")

    @staticmethod
    def _normalize_datetime(df: pd.DataFrame) -> pd.DataFrame:
        out = df.reset_index(drop=True).copy()
        if "datetime" not in out.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                out.insert(0, "datetime", df.index.to_series().values)
            else:
                out.insert(0, "datetime", np.arange(len(out)))
        return out


__all__ = ["V12PhysicsEnv", "V12Action", "SimPosition"]
