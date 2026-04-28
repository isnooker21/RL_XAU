"""
v12_brain_ppo.py
================
Train V12 Genesis with **MaskablePPO** + **LSTM** feature extractor (stacked obs).

Dependencies:
  pip install gymnasium numpy pandas torch stable-baselines3 sb3-contrib onnx

Example:
  python v12_brain_ppo.py --steps 200000 --data XAUUSD_M5_wetrade.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from v12_config import CONFIG  # noqa: E402
from v12_physics_env import V12PhysicsEnv  # noqa: E402

N_STACK = 8
N_ENVS = 4
N_STEPS = 512
BATCH_SIZE = 256
N_EPOCHS = 10
GAMMA = 0.99
LR = 2.5e-4
ENT_COEF = 0.01  # tune v4: กลาง run2 (0.015) กับ run3 (0.007)


class V12LstmExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        n_stack: int = N_STACK,
        n_features_per_step: int = CONFIG.n_obs_features,
        lstm_hidden: int = 128,
    ):
        super().__init__(observation_space, features_dim=lstm_hidden)
        expected = n_stack * n_features_per_step
        actual = int(np.prod(observation_space.shape))
        if expected != actual:
            raise ValueError(f"obs dim mismatch: expected {expected}, got {actual}")
        self.n_stack = n_stack
        self.n_features_per_step = n_features_per_step
        self.lstm = nn.LSTM(
            input_size=n_features_per_step,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.Linear(lstm_hidden, lstm_hidden), nn.LayerNorm(lstm_hidden), nn.ReLU())

    def forward(self, obs: th.Tensor) -> th.Tensor:
        b = obs.shape[0]
        x = obs.view(b, self.n_stack, self.n_features_per_step)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def policy_kwargs(n_features: int = CONFIG.n_obs_features):
    return dict(
        features_extractor_class=V12LstmExtractor,
        features_extractor_kwargs=dict(
            n_stack=N_STACK,
            n_features_per_step=n_features,
            lstm_hidden=128,
        ),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )


def load_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Loading: {path}")
    df = pd.read_csv(path, sep=None, engine="python")
    df = df.rename(columns={c: c.strip().lower().replace("<", "").replace(">", "") for c in df.columns})
    if {"date", "time"}.issubset(df.columns):
        df["datetime"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce"
        )
    elif "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["time"], errors="coerce")
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        raise ValueError(f"No datetime. Columns: {list(df.columns)}")

    if "tickvol" in df.columns and "volume" not in df.columns:
        df["volume"] = df["tickvol"]
    elif "tick_volume" in df.columns and "volume" not in df.columns:
        df["volume"] = df["tick_volume"]

    req = ["datetime", "open", "high", "low", "close", "volume"]
    keep = req + (["spread"] if "spread" in df.columns else [])
    df = df[keep].dropna(subset=["datetime"]).copy()
    if "spread" in df.columns:
        spread_med = float(df["spread"].median())
        if spread_med > 2.0:
            df["spread"] = df["spread"] * 0.01
    df = df.sort_values("datetime").reset_index(drop=True)
    df.index = pd.DatetimeIndex(df["datetime"])
    df.index.name = "ts"

    split = "2026-01-01"
    train_df = df[df["datetime"] < split].copy()
    test_df = df[df["datetime"] >= split].copy()
    if len(train_df) < 500:
        train_df = df.iloc[: max(len(df) - 200, 500)].copy()
        test_df = df.iloc[max(len(df) - 200, 500) :].copy()
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")
    return train_df, test_df


def parse_args():
    p = argparse.ArgumentParser(description="V12 MaskablePPO + LSTM trainer")
    p.add_argument("--data", type=str, default=os.path.join(BASE_DIR, "XAUUSD_M5_wetrade.csv"))
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=N_ENVS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=str, default=os.path.join(BASE_DIR, "v12_maskable_ppo_lstm"))
    p.add_argument("--mc-paths", type=int, default=None, help="Override MC paths per step (default: cfg.mc_n_paths_train)")
    p.add_argument("--no-subproc", action="store_true", help="Single DummyVecEnv (debug)")
    p.add_argument(
        "--ent-coef",
        type=float,
        default=ENT_COEF,
        help="Entropy coefficient (exploration). Default tuned for v2.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = CONFIG
    train_df, _ = load_data(args.data)

    n_envs = 1 if args.no_subproc else args.n_envs

    def make_one(seed_off: int):
        def _mk():
            env = V12PhysicsEnv(
                train_df,
                config=cfg,
                seed=args.seed + seed_off,
                random_start=True,
                mc_n_paths=args.mc_paths,
            )
            return ActionMasker(env, lambda e: e.action_masks())

        set_random_seed(args.seed + seed_off)
        return _mk

    if n_envs <= 1:
        vec = DummyVecEnv([make_one(0)])
    else:
        vec = SubprocVecEnv([make_one(i) for i in range(n_envs)])
    vec = VecMonitor(vec)
    vec = VecFrameStack(vec, n_stack=N_STACK)

    model = MaskablePPO(
        "MlpPolicy",
        vec,
        learning_rate=LR,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        ent_coef=args.ent_coef,
        verbose=1,
        seed=args.seed,
        policy_kwargs=policy_kwargs(cfg.n_obs_features),
        tensorboard_log=os.path.join(BASE_DIR, "v12_tb"),
    )
    ckpt = CheckpointCallback(save_freq=max(50_000 // n_envs, 1000), save_path=args.save + "_ckpt", name_prefix="v12")
    model.learn(total_timesteps=args.steps, callback=ckpt, progress_bar=True)
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    model.save(args.save)
    print(f"Saved: {args.save}.zip")


if __name__ == "__main__":
    main()
