"""
v12_executor.py
===============
Bridge between trained **MaskablePPO (V12)** and MT5 / external stack.

- **CSV bridge**: write `v12_command.csv` (action id), read optional `v12_state.csv`.
- **ONNX export**: policy logits [B, 3] for Stay/Buy/Sell (apply masks in EA).

Export example:
  python v12_executor.py export --model v12_maskable_ppo_lstm.zip --onnx v12_policy.onnx
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import warnings
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Stacked observation (must match training: VecFrameStack N_STACK=8)
N_STACK = 8
N_FEATURES = 64  # CONFIG.n_obs_features — duplicated here for ONNX spec without importing torch in EA
ACTION_NAMES = ["Stay", "Buy", "Sell"]


class PolicyLogitsWrapper(nn.Module):
    """SB3 MaskablePPO MlpPolicy → single Categorical logits (3 actions)."""

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feats = self.policy.extract_features(obs)
        latent_pi = self.policy.mlp_extractor.forward_actor(feats)
        return self.policy.action_net(latent_pi)


def export_onnx(
    model_zip: str,
    onnx_path: str,
    spec_path: Optional[str] = None,
    opset: int = 17,
) -> None:
    """Load MaskablePPO and export policy logits to ONNX."""
    from v12_brain_ppo import V12LstmExtractor  # noqa: F401 — register for load

    model = MaskablePPO.load(model_zip)
    wrapped = PolicyLogitsWrapper(model.policy)
    wrapped.eval()
    obs_dim = N_STACK * N_FEATURES
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    torch.onnx.export(
        wrapped,
        dummy,
        onnx_path,
        input_names=["obs"],
        output_names=["action_logits"],
        opset_version=opset,
        dynamic_axes={"obs": {0: "batch"}, "action_logits": {0: "batch"}},
    )
    if spec_path:
        spec = {
            "obs_dim": obs_dim,
            "n_stack": N_STACK,
            "n_features_per_step": N_FEATURES,
            "action_dim": 3,
            "action_names": ACTION_NAMES,
            "notes": "Stack order: oldest frame first (SB3 VecFrameStack convention).",
        }
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2)
    print(f"ONNX written: {onnx_path}")


class V12CsvBridge:
    """Minimal file bridge for MQL5 `FileRead`/`FileWrite` integration."""

    def __init__(
        self,
        data_dir: str,
        command_name: str = "v12_command.csv",
        state_name: str = "v12_state.csv",
    ):
        self.data_dir = data_dir
        self.cmd_path = os.path.join(data_dir, command_name)
        self.state_path = os.path.join(data_dir, state_name)

    def write_action(self, action_id: int, meta: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.cmd_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            row = [int(time.time() * 1000), int(action_id)]
            if meta:
                row.append(json.dumps(meta, separators=(",", ":")))
            w.writerow(row)

    def read_state(self) -> Optional[Dict[str, float]]:
        if not os.path.isfile(self.state_path):
            return None
        with open(self.state_path, newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            row = next(iter(r), None)
        if not row:
            return None
        keys = ["bid", "ask", "equity", "margin_level", "net_lots"]
        out: Dict[str, float] = {}
        for i, k in enumerate(keys):
            if i < len(row):
                try:
                    out[k] = float(row[i])
                except ValueError:
                    continue
        return out


def parse_args():
    p = argparse.ArgumentParser(description="V12 executor / ONNX export")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="Export policy to ONNX")
    e.add_argument("--model", type=str, default=os.path.join(BASE_DIR, "v12_maskable_ppo_lstm.zip"))
    e.add_argument("--onnx", type=str, default=os.path.join(BASE_DIR, "v12_policy.onnx"))
    e.add_argument("--spec", type=str, default=os.path.join(BASE_DIR, "v12_onnx_spec.json"))

    b = sub.add_parser("bridge-demo", help="Write a sample command row")
    b.add_argument("--dir", type=str, default=BASE_DIR)
    b.add_argument("--action", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.cmd == "export":
        export_onnx(args.model, args.onnx, spec_path=args.spec)
    elif args.cmd == "bridge-demo":
        V12CsvBridge(args.dir).write_action(args.action, meta={"source": "v12_executor"})


if __name__ == "__main__":
    main()
