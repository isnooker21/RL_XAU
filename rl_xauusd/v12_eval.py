"""
v12_eval.py
===========
Evaluate a trained V12 MaskablePPO checkpoint on **test** (or train) split.

Uses the same VecFrameStack + ActionMasker layout as training so obs dims match.

Example:
  python v12_eval.py --model v12_maskable_ppo_lstm.zip --episodes 5 --max-steps 25000
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from v12_brain_ppo import N_STACK, load_data, policy_kwargs  # noqa: E402
from v12_config import CONFIG  # noqa: E402
from v12_physics_env import V12PhysicsEnv  # noqa: E402


def _inner_action_masker(vec_env):
    """Unwrap VecFrameStack → VecMonitor → DummyVecEnv → ActionMasker."""
    v = vec_env
    while hasattr(v, "venv"):
        v = v.venv
    return v.envs[0]


def build_eval_vec(df, seed: int, mc_paths: Optional[int], random_start: bool):
    def _mk():
        env = V12PhysicsEnv(
            df,
            config=CONFIG,
            seed=seed,
            random_start=random_start,
            mc_n_paths=mc_paths,
        )
        return ActionMasker(env, lambda e: e.action_masks())

    set_random_seed(seed)
    vec = DummyVecEnv([_mk])
    vec = VecMonitor(vec)
    vec = VecFrameStack(vec, n_stack=N_STACK)
    return vec


def run_episode(vec, model: MaskablePPO, max_steps: int) -> dict:
    obs = vec.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    total_r = 0.0
    steps = 0
    min_margin = float("inf")
    n_margin_viol = 0
    last_eq = CONFIG.initial_capital
    inner = _inner_action_masker(vec)
    n_stay = n_buy = n_sell = 0

    for _ in range(max_steps):
        masks = np.array([inner.action_masks()], dtype=bool)
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        a = int(action[0]) if hasattr(action, "__len__") else int(action)
        if a == 0:
            n_stay += 1
        elif a == 1:
            n_buy += 1
        else:
            n_sell += 1
        obs, rewards, dones, infos = vec.step(action)

        total_r += float(rewards[0])
        steps += 1
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        if isinstance(info, list) and len(info) > 0:
            info = info[-1]
        ml = None
        eq = None
        if isinstance(info, dict):
            ml = info.get("margin_level")
            eq = info.get("equity")
        if ml is None and hasattr(inner, "env"):
            ml = getattr(inner.env, "_margin_level", None)
        if eq is None and hasattr(inner, "env"):
            eq = getattr(inner.env, "_equity", None)
        if ml is not None:
            ml = float(ml)
            min_margin = min(min_margin, ml)
            if ml < CONFIG.risk.margin_level_min_open:
                n_margin_viol += 1
        if eq is not None:
            last_eq = float(eq)
        if dones[0]:
            break

    return dict(
        reward=total_r,
        steps=steps,
        min_margin_level=min_margin if min_margin < float("inf") else 0.0,
        margin_below_guard_steps=n_margin_viol,
        final_equity=last_eq,
        n_stay=n_stay,
        n_buy=n_buy,
        n_sell=n_sell,
    )


def main():
    p = argparse.ArgumentParser(description="V12 policy evaluation")
    p.add_argument("--model", type=str, default=os.path.join(BASE_DIR, "v12_maskable_ppo_lstm.zip"))
    p.add_argument("--data", type=str, default=os.path.join(BASE_DIR, "XAUUSD_M5_wetrade.csv"))
    p.add_argument("--split", choices=("test", "train"), default="test")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=30_000, help="Cap steps per episode")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--mc-paths", type=int, default=2000, help="MC paths per step (speed)")
    args = p.parse_args()

    import v12_brain_ppo  # noqa: F401 — registers LSTM for zip load

    train_df, test_df = load_data(args.data)
    df = test_df if args.split == "test" else train_df
    if len(df) < CONFIG.lookback_bars + 50:
        print("Split too short; using full dataframe.")
        _, df = load_data(args.data)
        df = df.iloc[-(CONFIG.lookback_bars + 500) :].copy()

    print(f"Loading model: {args.model}")
    model = MaskablePPO.load(args.model)

    rng = np.random.default_rng(args.seed)
    agg = []
    for ep in range(args.episodes):
        vec = build_eval_vec(
            df,
            seed=int(rng.integers(0, 1_000_000)),
            mc_paths=args.mc_paths,
            random_start=True,
        )
        stats = run_episode(vec, model, args.max_steps)
        stats["episode"] = ep
        agg.append(stats)
        vec.close()
        print(
            f"  ep {ep + 1}/{args.episodes}  reward={stats['reward']:.4f}  "
            f"steps={stats['steps']}  min_margin%={stats['min_margin_level']:.2f}  "
            f"viol_steps={stats['margin_below_guard_steps']}  equity={stats['final_equity']:.2f}  "
            f"act S/B/S={stats['n_stay']}/{stats['n_buy']}/{stats['n_sell']}"
        )

    rewards = [a["reward"] for a in agg]
    mins = [a["min_margin_level"] for a in agg]
    tb = sum(a["n_buy"] for a in agg)
    ts = sum(a["n_sell"] for a in agg)
    tt = sum(a["n_stay"] for a in agg)
    print("---")
    print(f"mean_reward={float(np.mean(rewards)):.4f}  std={float(np.std(rewards)):.4f}")
    print(f"mean_min_margin_level%={float(np.mean(mins)):.2f}  worst_ep_min={float(np.min(mins)):.2f}")
    print(f"total_actions  Stay={tt}  Buy={tb}  Sell={ts}")


if __name__ == "__main__":
    main()
