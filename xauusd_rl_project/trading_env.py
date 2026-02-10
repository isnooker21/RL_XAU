"""
Custom Trading Environment for XAUUSD RL Training
==================================================
Gym Environment for Multi-Timeframe Trading with Stable Baselines3

Features:
- Multi-timeframe observations (M15, H1, H4)
- Actions: Buy, Sell, Close, Hold
- Reward: Net Profit + Equity Change
- Position Management with proper risk
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """
    Custom Trading Environment for XAUUSD
    
    Action Space:
        0: Hold (ไม่ทำอะไร)
        1: Buy (เปิด Long)
        2: Sell (เปิด Short)
        3: Close Position (ปิดออเดอร์)
    
    Observation Space:
        - Price data (OHLC for M15, H1, H4)
        - Technical indicators (RSI, ATR, MACD, etc.)
        - Position state (direction, profit, holding_time)
    
    Reward:
        - Net Profit from closed trades
        - Equity change from open position
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 df: pd.DataFrame,
                 initial_balance: float = 10000.0,
                 lot_size: float = 0.01,
                 commission: float = 0.0,
                 leverage: int = 100,
                 lookback_window: int = 50,
                 max_drawdown_pct: float = 0.15,
                 slippage_range: tuple = (5, 10),
                 rollover_spread_multiplier: float = 2.5):
        """
        Initialize Trading Environment
        
        Args:
            df: DataFrame with READY_TO_TRAIN data
            initial_balance: Starting capital (USD)
            lot_size: Trading size (0.01 = micro lot)
            commission: Commission per trade (USD)
            leverage: Trading leverage
            lookback_window: Number of past timesteps to observe
            max_drawdown_pct: Maximum drawdown before hard stop (default: 15%)
            slippage_range: Random slippage range in points (default: 5-10)
            rollover_spread_multiplier: Spread multiplier during rollover (default: 2.5x)
        """
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.lot_size = lot_size
        self.commission = commission
        self.leverage = leverage
        self.lookback_window = lookback_window
        self.max_drawdown_pct = max_drawdown_pct
        self.slippage_range = slippage_range
        self.rollover_spread_multiplier = rollover_spread_multiplier
        
        # Trading state
        self.current_step = 0
        self.balance = initial_balance
        self.equity = initial_balance
        self.position = 0  # 0: None, 1: Long, -1: Short
        self.entry_price = 0.0
        self.position_profit = 0.0
        self.holding_time = 0
        self.losing_holding_time = 0  # Time holding losing position
        
        # Episode stats
        self.total_profit = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.peak_equity = initial_balance
        self.equity_history = []  # Track equity for consistency bonus
        
        # Feature columns (exclude time)
        self.feature_columns = [col for col in df.columns if col != 'time']
        
        # Action space: 4 discrete actions
        self.action_space = spaces.Discrete(4)
        
        # Observation space: features + position state
        # Position state: [has_position, position_direction, normalized_profit, normalized_holding_time]
        n_features = len(self.feature_columns)
        n_position_features = 4
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features + n_position_features,),
            dtype=np.float32
        )
        
        logger.info(f"TradingEnv initialized:")
        logger.info(f"  Data: {len(self.df)} timesteps")
        logger.info(f"  Features: {n_features}")
        logger.info(f"  Observation shape: {self.observation_space.shape}")
        logger.info(f"  Action space: {self.action_space.n}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reset to random starting point (leave enough room for episode)
        max_start = len(self.df) - self.lookback_window - 500
        self.current_step = np.random.randint(self.lookback_window, max_start)
        
        # Reset trading state
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.position_profit = 0.0
        self.holding_time = 0
        self.losing_holding_time = 0
        
        # Reset episode stats
        self.total_profit = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.peak_equity = self.initial_balance
        self.equity_history = []
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in environment
        
        Args:
            action: 0=Hold, 1=Buy, 2=Sell, 3=Close
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        previous_equity = self.equity
        reward = 0.0
        
        current_price = self._get_current_price()
        
        # Execute action
        if action == 1:  # Buy
            reward += self._open_position(1, current_price)
        elif action == 2:  # Sell
            reward += self._open_position(-1, current_price)
        elif action == 3:  # Close
            reward += self._close_position(current_price)
        # action == 0: Hold (ไม่ทำอะไร)
        
        # Update position profit if holding
        if self.position != 0:
            self.position_profit = self._calculate_profit(current_price)
            self.holding_time += 1
            self.equity = self.balance + self.position_profit
            
            # Track if holding losing position
            if self.position_profit < 0:
                self.losing_holding_time += 1
            else:
                self.losing_holding_time = 0
            
            # Update peak equity
            if self.equity > self.peak_equity:
                self.peak_equity = self.equity
            
            # Basic reward from equity change
            equity_change = self.equity - previous_equity
            reward += equity_change / self.initial_balance  # Normalized
        
        # Track equity history for consistency bonus
        self.equity_history.append(self.equity)
        
        # Calculate advanced reward components
        reward += self._calculate_advanced_reward()
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        terminated = False
        truncated = False
        
        # Episode ends if:
        # 1. Reached end of data
        if self.current_step >= len(self.df) - 1:
            truncated = True
            # Close any open position
            if self.position != 0:
                reward += self._close_position(current_price)
        
        # 2. HARD STOP: Drawdown exceeds maximum (15%)
        current_drawdown = self._calculate_drawdown()
        if current_drawdown > self.max_drawdown_pct:
            terminated = True
            # Exponential penalty for excessive drawdown
            drawdown_penalty = -np.exp(current_drawdown * 10) / self.initial_balance
            reward += drawdown_penalty
            
            if self.position != 0:
                reward += self._close_position(current_price)
        
        # 3. Equity drops too low (margin call)
        elif self.equity < self.initial_balance * 0.3:  # Lost 70%
            terminated = True
            if self.position != 0:
                reward += self._close_position(current_price)
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _open_position(self, direction: int, price: float) -> float:
        """
        Open new position (close existing if any)
        
        Args:
            direction: 1=Long, -1=Short
            price: Entry price
            
        Returns:
            reward from closing previous position (if any)
        """
        reward = 0.0
        
        # Close existing position first
        if self.position != 0:
            reward = self._close_position(price)
        
        # Apply slippage to entry price
        is_buy = (direction == 1)
        slipped_price = self._apply_slippage(price, is_buy)
        
        # Apply spread (widen entry price against trader)
        spread = self._get_current_spread()
        if is_buy:
            slipped_price += spread / 2  # Buy at ask
        else:
            slipped_price -= spread / 2  # Sell at bid
        
        # Open new position
        self.position = direction
        self.entry_price = slipped_price
        self.holding_time = 0
        self.losing_holding_time = 0
        self.position_profit = 0.0
        
        # Deduct commission
        self.balance -= self.commission
        
        return reward
    
    def _close_position(self, price: float) -> float:
        """
        Close current position
        
        Args:
            price: Exit price
            
        Returns:
            reward from trade
        """
        if self.position == 0:
            return 0.0
        
        # Apply slippage to exit price
        is_buy = (self.position == -1)  # Closing short = buying, closing long = selling
        slipped_price = self._apply_slippage(price, is_buy)
        
        # Apply spread (widen exit price against trader)
        spread = self._get_current_spread()
        if self.position == 1:  # Closing long = sell at bid
            slipped_price -= spread / 2
        else:  # Closing short = buy at ask
            slipped_price += spread / 2
        
        # Calculate profit with slippage
        profit = self._calculate_profit(slipped_price)
        
        # Update balance
        self.balance += profit
        self.balance -= self.commission  # Exit commission
        self.equity = self.balance
        
        # Update peak equity
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        
        # Update stats
        self.total_profit += profit
        self.total_trades += 1
        if profit > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Reset position
        self.position = 0
        self.entry_price = 0.0
        self.position_profit = 0.0
        self.holding_time = 0
        self.losing_holding_time = 0
        
        # Reward = normalized profit
        reward = profit / self.initial_balance
        
        return reward
    
    def _calculate_profit(self, current_price: float) -> float:
        """Calculate profit/loss of current position"""
        if self.position == 0:
            return 0.0
        
        # Price difference
        price_diff = (current_price - self.entry_price) * self.position
        
        # Profit = price_diff * lot_size * contract_size
        # For XAUUSD: 1 lot = 100 oz
        contract_size = 100
        profit = price_diff * self.lot_size * contract_size
        
        return profit
    
    def _get_current_price(self) -> float:
        """Get current M15 close price"""
        return self.df.iloc[self.current_step]['close_M15']
    
    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """
        Apply random slippage to price
        
        Args:
            price: Original price
            is_buy: True if buying, False if selling
            
        Returns:
            Price with slippage applied
        """
        # Random slippage in range (e.g., 5-10 points = 0.05-0.10 USD for XAUUSD)
        slippage_points = np.random.uniform(self.slippage_range[0], self.slippage_range[1])
        slippage = slippage_points / 10  # Convert points to price
        
        # Buy: slippage increases price, Sell: slippage decreases price
        if is_buy:
            return price + slippage
        else:
            return price - slippage
    
    def _get_current_spread(self) -> float:
        """
        Get current spread (varies during rollover hours)
        
        Returns:
            Spread in USD
        """
        # Base spread for XAUUSD (e.g., 0.3-0.5 USD)
        base_spread = 0.4
        
        # Check if in rollover period (around 21:00-01:00 UTC)
        # For simplicity, use modulo on step count
        if 'time' in self.df.columns:
            current_time = self.df.iloc[self.current_step]['time']
            hour = pd.to_datetime(current_time).hour
            
            # Rollover hours: 21:00-01:00 UTC (widen spread)
            if hour >= 21 or hour <= 1:
                return base_spread * self.rollover_spread_multiplier
        
        return base_spread
    
    def _calculate_drawdown(self) -> float:
        """
        Calculate current drawdown percentage
        
        Returns:
            Drawdown as percentage (0.0 to 1.0)
        """
        if self.peak_equity == 0:
            return 0.0
        
        drawdown = (self.peak_equity - self.equity) / self.peak_equity
        return max(0.0, drawdown)
    
    def _calculate_advanced_reward(self) -> float:
        """
        Calculate advanced reward components:
        1. Time-based penalty for holding losing positions
        2. Profit consistency bonus (low equity volatility)
        
        Returns:
            Additional reward/penalty
        """
        additional_reward = 0.0
        
        # 1. Time-based Penalty: Penalize holding losing positions
        if self.position != 0 and self.position_profit < 0:
            # Exponentially increasing penalty over time
            time_penalty = -(self.losing_holding_time ** 1.5) / 10000
            additional_reward += time_penalty
        
        # 2. Profit Consistency Bonus: Reward smooth equity curve
        if len(self.equity_history) >= 20:  # Need enough history
            recent_equity = self.equity_history[-20:]
            
            # Calculate volatility of recent equity changes
            equity_returns = np.diff(recent_equity) / (np.array(recent_equity[:-1]) + 1e-10)
            equity_volatility = np.std(equity_returns)
            
            # Check if equity is generally increasing
            equity_trend = (recent_equity[-1] - recent_equity[0]) / (self.initial_balance + 1e-10)
            
            # Bonus if equity grows steadily with low volatility
            if equity_trend > 0 and equity_volatility < 0.02:  # Low vol + positive trend
                consistency_bonus = equity_trend * 0.1  # Small bonus
                additional_reward += consistency_bonus
        
        return additional_reward
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation
        
        Returns:
            Array of [market_features, position_features]
        """
        # Market features from current timestep
        current_row = self.df.iloc[self.current_step]
        market_features = current_row[self.feature_columns].values.astype(np.float32)
        
        # Position features
        has_position = 1.0 if self.position != 0 else 0.0
        position_direction = float(self.position)  # -1, 0, or 1
        normalized_profit = self.position_profit / self.initial_balance if self.position != 0 else 0.0
        normalized_holding_time = self.holding_time / 100.0  # Normalize by typical holding period
        
        position_features = np.array([
            has_position,
            position_direction,
            normalized_profit,
            normalized_holding_time
        ], dtype=np.float32)
        
        # Concatenate
        obs = np.concatenate([market_features, position_features])
        
        return obs
    
    def _get_info(self) -> Dict:
        """Get additional info for logging"""
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        current_drawdown = self._calculate_drawdown()
        
        return {
            'step': self.current_step,
            'balance': self.balance,
            'equity': self.equity,
            'position': self.position,
            'position_profit': self.position_profit,
            'total_profit': self.total_profit,
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'drawdown': current_drawdown,
            'peak_equity': self.peak_equity,
        }
    
    def render(self, mode='human'):
        """Render environment state"""
        info = self._get_info()
        print(f"\nStep: {info['step']}")
        print(f"Balance: ${info['balance']:.2f}")
        print(f"Equity: ${info['equity']:.2f}")
        print(f"Position: {info['position']} (Profit: ${info['position_profit']:.2f})")
        print(f"Total Trades: {info['total_trades']} (Win Rate: {info['win_rate']:.1%})")

