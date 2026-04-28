"""
v12_config.py
=============
Central parameters for V12 Genesis (physics-based RL, XAUUSD / WeTrade).

Broker assumptions (tune to live feed):
  - Spread: ~35–40 points (0.35–0.40 USD on typical XAUUSD quoting).
  - Leverage: 1:500–1:1000.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class BrokerSpec:
    """
    WeTrade / XAUUSD — ปรับให้ตรงไลฟ์ตามลำดับนี้:

    1) **Contract / PnL**
       MT5 → คลิกขวาสัญลักษณ์ → Specification: `Contract size`, `Tick size`, `Profit mode`.
       ตั้ง `contract_size` ให้เท่าค่า “ต่อ 1.00 lot” ใน spec (ทองมัก 100 oz/lot แต่โบรกอาจต่าง).

    2) **Point / spread เป็นราคา**
       `full_spread_price = spread_points * point_size` ต้องเท่าความกว้าง bid–ask จริงในหน่วยราคา
       (เช่น 37 points × 0.01 = 0.37 ถ้า 1 point = 0.01).
       ตรวจ: ใน MT5 spread แสดงเป็น points — เทียบกับ `_Digits` ว่า 1 point กี่โควต.

    3) **Commission**
       ใน env คิดแบบ **USD ต่อ 1.0 lot ต่อครั้งที่เปิด/ปิดบางส่วน** (ไม่ใช่ round-turn อัตโนมัติ).
       ถ้าโบรกเก็บ “รอบเทรดละ X” ให้แบ่งใส่ (เช่น เปิดขาเดียว = X/2) ให้ตรงกับสิ่งที่คุณอยากให้ sim หักต่อไม้.

    4) **Margin / margin level**
       สูตร sim: `used_margin ≈ Σ |lot| × price × contract_size / leverage` (กรณี netting แบบ gross ต่อขา).
       MT5 อาจใช้ **hedged margin** หรือโมเดลอื่น → เปิด 1.00 lot แล้วดู `AccountInfoDouble(MARGIN)` /
       `OrderCalcMargin` เปรียบเทียบกับค่าที่ sim ได้ แล้วปรับ `leverage` หรือ scale factor จนใกล้
       (หรือจะเพิ่มฟิลด์ margin override ในโค้ดทีหลังก็ได้).

    5) **Swap** — ยังไม่มีใน V12 env; ไลฟ์สวอปต้องรับรู้แยกหรือขยาย env.
    """

    symbol: str = "XAUUSD"
    # Point = smallest price increment in your data (often 0.01 for gold).
    point_size: float = 0.01
    pip_size: float = 0.10
    contract_size: float = 100.0  # oz per 1.0 lot — ต้องตรง MT5 Specification
    leverage: float = 500.0  # ต้องตรงเลเวอเรจบัญชีจริง / สัญลักษณ์
    leverage_max: float = 1000.0
    # Spread in *points* (not pips); 35–40 per user spec
    spread_points: float = 37.5
    spread_points_min: float = 35.0
    spread_points_max: float = 40.0
    # USD per 1.0 lot per charge event (open / partial close leg); tune to ticket history
    commission_per_lot: float = 0.0


@dataclass(frozen=True)
class PhysicsSpec:
    """Discrete-time kinematics on log-price or price — env chooses scale."""

    use_log_price: bool = True
    velocity_window: int = 3
    acceleration_window: int = 5
    # Kinetic energy proxy: 0.5 * m * v^2 with m=1 (dimensionless mass)
    kinetic_mass: float = 1.0
    # Smoothing for finite differences (optional EMA alpha in env)
    ema_alpha_velocity: float = 0.35


@dataclass(frozen=True)
class HurstSpec:
    """Regime detection: H < 0.5 mean-reverting, H > 0.5 trending (heuristic)."""

    window: int = 256
    min_window: int = 64
    # Labels used inside observation / reward shaping
    trending_threshold: float = 0.55
    mean_revert_threshold: float = 0.45


@dataclass(frozen=True)
class ZScoreSpec:
    """Anti-Doi: mask Buys when price high vs mean; mask Sells when price low."""

    lookback: int = 96
    clip_sigma: float = 4.0
    # |z| above this → start masking directional adds
    buy_mask_z: float = 1.5   # block Buy when z >= this (overextended up)
    sell_mask_z: float = -1.5  # block Sell when z <= this (stretched down)


@dataclass(frozen=True)
class MonteCarloSpec:
    """Path simulation for confidence and margin-at-risk (full engine in v12_monte_carlo)."""

    n_paths: int = 10_000
    horizon_steps: int = 32
    dt_fraction_of_bar: float = 1.0  # one bar = one step in GBM discretization
    stress_spread_mult_range: Tuple[float, float] = (1.0, 2.5)
    stress_jump_sigma_mult: float = 3.0
    random_seed_offset: int = 12_012


@dataclass(frozen=True)
class RiskSpec:
    """Hard guardrails and drawdown envelope."""

    margin_level_min_open: float = 100.0  # HARD: no new Buy/Sell below this
    margin_level_warning: float = 150.0
    target_max_drawdown_pct: float = 0.225  # 20–25% band, center 22.5%
    target_trades_per_day_min: float = 1.5
    target_trades_per_day_max: float = 3.0
    # Delta-neutral system hedge: when margin weak and MC tail is bad
    mc_hedge_margin_level_max: float = 180.0  # only consider auto-hedge below this
    mc_hedge_mar_threshold: float = 0.015  # margin_at_risk from MC tail


@dataclass(frozen=True)
class RewardSpec:
    """Reward shaping anchors (env scales to stable magnitude)."""

    scale_realized_pnl: float = 1.0
    margin_penalty_below_guard: float = 0.08  # tune v4: กลางๆ ระหว่าง v1 กับ v3 (run3 นิ่งเกิน)
    doi_penalty_weight: float = 0.02  # extra penalty when acting against z-mask intent
    chaos_survive_bonus: float = 0.001  # tiny bonus if margin OK after adversarial shock
    trade_open_penalty: float = 0.00055  # tune v4: balance ระหว่าง run2 (spam) กับ run3 (แทบไม่เทรด)
    tpd_target_mid: float = 2.25  # (1.5 + 3.0) / 2
    tpd_shaping_weight: float = 0.005  # tune v2: was 0.015 — TPD นุ่มขึ้น
    tpd_shaping_max_err: float = 0.35  # cap |tpd error| กันบทลงโทษตอน tpd≈0 ทุก step หนักเกิน
    margin_bonus_per_pct_above_guard: float = 0.00035  # tune v2: was 0.001 — ลดโบนัส margin สูงตอนไม่มี lot
    drawdown_penalty_weight: float = 0.05  # vs peak equity within episode


@dataclass(frozen=True)
class AdversarialSpec:
    """Training-time chaos: spikes / spread widening."""

    enabled: bool = True
    chaos_prob_per_step: float = 0.02
    spike_sigma_mult_range: Tuple[float, float] = (2.0, 5.0)
    spread_widen_mult_range: Tuple[float, float] = (1.2, 2.0)


@dataclass(frozen=True)
class GridSpec:
    """Dynamic grid step (env implements expansion/contraction from a, vol)."""

    base_distance_pips: float = 15.0
    min_distance_pips: float = 8.0
    max_distance_pips: float = 45.0
    vol_lookback: int = 32


@dataclass(frozen=True)
class SimSpec:
    """In-env execution / recycling (simulation)."""

    min_lot: float = 0.01
    lot_step: float = 0.01
    max_open_positions: int = 40
    max_abs_net_lots: float = 0.22  # cap aggregate directional exposure
    emergency_deleverage_margin_level: float = 120.0  # start forced trim below this
    emergency_close_lot: float = 0.02  # forced close chunk per step when stressed
    # Houdini: use session profit buffer to chip worst underwater legs
    houdini_buffer_min: float = 15.0  # USD (account currency) before recycle
    houdini_close_lot: float = 0.01
    # Bars per day for TPD (M5 ≈ 288, M15 ≈ 96)
    bars_per_day: float = 288.0


@dataclass(frozen=True)
class V12Config:
    """Single bundle passed into the env and trainers."""

    broker: BrokerSpec = field(default_factory=BrokerSpec)
    physics: PhysicsSpec = field(default_factory=PhysicsSpec)
    hurst: HurstSpec = field(default_factory=HurstSpec)
    zscore: ZScoreSpec = field(default_factory=ZScoreSpec)
    monte_carlo: MonteCarloSpec = field(default_factory=MonteCarloSpec)
    risk: RiskSpec = field(default_factory=RiskSpec)
    reward: RewardSpec = field(default_factory=RewardSpec)
    adversarial: AdversarialSpec = field(default_factory=AdversarialSpec)
    grid: GridSpec = field(default_factory=GridSpec)
    sim: SimSpec = field(default_factory=SimSpec)

    # Account / episode defaults
    initial_capital: float = 10_000.0
    max_lot: float = 1.0
    lookback_bars: int = 128
    episode_max_steps: Optional[int] = None  # None = use data-driven window

    # RL spaces (recurrent policy expects fixed obs dim)
    n_obs_features: int = 64  # env packs physics + hurst + z + MC stats + book + regime flags
    # Faster training: optional override inside env ctor
    mc_n_paths_train: Optional[int] = 2000


# Default singleton for imports: `from v12_config import CONFIG`
CONFIG = V12Config()
