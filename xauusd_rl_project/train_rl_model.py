"""
RL Model Training Script for XAUUSD Trading
============================================
Train PPO agent using Stable Baselines3

Features:
- PPO Algorithm
- TensorBoard logging
- Model checkpoints every 100,000 steps
- Custom callback for monitoring
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Custom environment
from trading_env import TradingEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class EarlyStoppingCallback(BaseCallback):
    """
    Custom Early Stopping Callback
    
    Stop training when validation reward doesn't improve for patience evaluations
    or when training loss starts diverging
    """
    def __init__(self, eval_callback=None, patience: int = 5, min_delta: float = 0.01, verbose: int = 1):
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.eval_callback = eval_callback
        self.patience = patience
        self.min_delta = min_delta
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
        self.prev_loss = None
        self.divergence_count = 0
        
    def _on_step(self) -> bool:
        """Called at each training step"""
        # Check for loss divergence every 1000 steps
        if self.n_calls % 1000 == 0:
            # Use value_loss instead of total loss (more stable indicator)
            if 'train/value_loss' in self.logger.name_to_value:
                current_loss = self.logger.name_to_value['train/value_loss']
                
                if self.prev_loss is not None and current_loss > 0 and self.prev_loss > 0:
                    # Check if loss increased significantly
                    if current_loss > self.prev_loss * 2.0:  # 100% increase
                        self.divergence_count += 1
                        if self.verbose > 0:
                            logger.warning(f"⚠️  Loss divergence detected: {self.prev_loss:.6f} -> {current_loss:.6f} ({self.divergence_count}/{self.patience})")
                        
                        if self.divergence_count >= self.patience:
                            if self.verbose > 0:
                                logger.warning("🛑 Early stopping: Loss diverged!")
                            return False
                    else:
                        self.divergence_count = 0  # Reset if loss is stable
                
                if current_loss > 0:  # Only track positive losses
                    self.prev_loss = current_loss
        
        # Check validation performance (triggered by EvalCallback)
        if self.eval_callback is not None and hasattr(self.eval_callback, 'last_mean_reward'):
            mean_reward = self.eval_callback.last_mean_reward
            
            # Only check after first evaluation
            if mean_reward != -np.inf:
                if mean_reward > self.best_mean_reward + self.min_delta:
                    self.best_mean_reward = mean_reward
                    self.no_improvement_count = 0
                    if self.verbose > 0:
                        logger.info(f"✅ Validation improved: {mean_reward:.4f}")
                else:
                    self.no_improvement_count += 1
                    if self.verbose > 0:
                        logger.warning(f"⚠️  No improvement for {self.no_improvement_count}/{self.patience} evaluations (current: {mean_reward:.4f}, best: {self.best_mean_reward:.4f})")
                    
                    if self.no_improvement_count >= self.patience:
                        if self.verbose > 0:
                            logger.warning("🛑 Early stopping: No validation improvement!")
                        return False
        
        return True


class TrainingConfig:
    """Training Configuration"""
    
    # Data
    DATA_FILE = "processed_data/XAUUSD_READY_TO_TRAIN.parquet"
    
    # Environment
    INITIAL_BALANCE = 10000.0
    LOT_SIZE = 0.01
    COMMISSION = 0.0  # No commission for now
    LEVERAGE = 100
    LOOKBACK_WINDOW = 50
    
    # Training
    TOTAL_TIMESTEPS = 1_000_000  # 1M steps for first training
    LEARNING_RATE = 3e-4
    N_STEPS = 2048  # Steps per update
    BATCH_SIZE = 64
    N_EPOCHS = 10
    GAMMA = 0.99  # Discount factor
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    ENT_COEF = 0.01  # Entropy coefficient (exploration)
    VF_COEF = 0.5  # Value function coefficient
    MAX_GRAD_NORM = 0.5
    
    # Checkpoints
    CHECKPOINT_FREQ = 100_000  # Save every 100k steps
    CHECKPOINT_DIR = "models/xauusd_model_checkpoints"
    
    # Logging
    TENSORBOARD_LOG = "logs/tensorboard"
    
    # Train/Validation/Test Split
    TRAIN_RATIO = 0.70  # 70% for training
    VAL_RATIO = 0.10    # 10% for validation
    TEST_RATIO = 0.20   # 20% for hidden test (2025-2026) - NEVER used in training
    
    # Early Stopping
    EARLY_STOP_PATIENCE = 5  # Stop if val performance doesn't improve for 5 evaluations
    EARLY_STOP_MIN_DELTA = 0.01  # Minimum improvement to count as better


def load_data(file_path: str) -> pd.DataFrame:
    """Load READY_TO_TRAIN data"""
    logger.info(f"Loading data from: {file_path}")
    
    df = pd.read_parquet(file_path)
    
    logger.info(f"Data loaded: {len(df):,} rows, {len(df.columns)} columns")
    logger.info(f"Date range: {df['time'].min()} to {df['time'].max()}")
    
    return df


def split_train_val_test(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.1) -> tuple:
    """
    Split data into train, validation, and hidden test sets (chronological)
    
    Args:
        df: Full dataset
        train_ratio: Ratio for training (0.7 = 70%)
        val_ratio: Ratio for validation (0.1 = 10%)
        
    Returns:
        (train_df, val_df, test_df)
        
    Note:
        - Test set (last 20%) is HIDDEN and NEVER used during training
        - Only used for final evaluation after training completes
    """
    train_end_idx = int(len(df) * train_ratio)
    val_end_idx = int(len(df) * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end_idx].reset_index(drop=True)
    val_df = df.iloc[train_end_idx:val_end_idx].reset_index(drop=True)
    test_df = df.iloc[val_end_idx:].reset_index(drop=True)
    
    logger.info(f"Train set:      {len(train_df):,} rows ({df['time'].iloc[0]} to {df['time'].iloc[train_end_idx-1]})")
    logger.info(f"Validation set: {len(val_df):,} rows ({df['time'].iloc[train_end_idx]} to {df['time'].iloc[val_end_idx-1]})")
    logger.info(f"Test set:       {len(test_df):,} rows ({df['time'].iloc[val_end_idx]} to {df['time'].iloc[-1]}) - HIDDEN")
    logger.warning("⚠️  Test set is HIDDEN and will NOT be used during training!")
    
    return train_df, val_df, test_df


def create_env(df: pd.DataFrame, config: TrainingConfig, is_eval: bool = False):
    """
    Create vectorized environment
    
    Args:
        df: DataFrame with trading data
        config: Training configuration
        is_eval: Whether this is evaluation environment
        
    Returns:
        Vectorized environment
    """
    def make_env():
        env = TradingEnv(
            df=df,
            initial_balance=config.INITIAL_BALANCE,
            lot_size=config.LOT_SIZE,
            commission=config.COMMISSION,
            leverage=config.LEVERAGE,
            lookback_window=config.LOOKBACK_WINDOW
        )
        # Wrap with Monitor for logging
        log_dir = "logs/monitor_eval" if is_eval else "logs/monitor_train"
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        env = Monitor(env, log_dir)
        return env
    
    # Create vectorized environment (single env for now)
    env = DummyVecEnv([make_env])
    
    # Normalize observations and rewards
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=config.GAMMA
    )
    
    return env


def create_callbacks(config: TrainingConfig, eval_env):
    """Create training callbacks with early stopping"""
    
    # 1. Checkpoint callback - save every 100k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=config.CHECKPOINT_FREQ,
        save_path=config.CHECKPOINT_DIR,
        name_prefix="xauusd_ppo",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    # 2. Evaluation callback - evaluate on VALIDATION set
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{config.CHECKPOINT_DIR}/best_model",
        log_path=f"{config.CHECKPOINT_DIR}/eval_logs",
        eval_freq=50_000,  # Evaluate every 50k steps
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    
    # 3. Early stopping callback (monitors eval_callback)
    early_stopping_callback = EarlyStoppingCallback(
        eval_callback=eval_callback,
        patience=config.EARLY_STOP_PATIENCE,
        min_delta=config.EARLY_STOP_MIN_DELTA,
        verbose=1
    )
    
    # Combine callbacks
    callback = CallbackList([checkpoint_callback, eval_callback, early_stopping_callback])
    
    return callback


def train_model(config: TrainingConfig):
    """Main training function"""
    
    logger.info("="*80)
    logger.info("STARTING RL TRAINING")
    logger.info("="*80)
    
    # 1. Load data
    df = load_data(config.DATA_FILE)
    logger.info("-"*80)
    
    # 2. Split train/validation/test
    train_df, val_df, test_df = split_train_val_test(df, config.TRAIN_RATIO, config.VAL_RATIO)
    logger.info("-"*80)
    
    # 3. Create environments (USE ONLY TRAIN AND VALIDATION)
    logger.info("Creating environments...")
    train_env = create_env(train_df, config, is_eval=False)
    eval_env = create_env(val_df, config, is_eval=True)  # Use VALIDATION set, NOT test
    logger.info("Environments created successfully")
    logger.info("⚠️  Test set is kept HIDDEN for final evaluation only!")
    logger.info("-"*80)
    
    # 4. Create directories
    Path(config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.TENSORBOARD_LOG).mkdir(parents=True, exist_ok=True)
    
    # 5. Create PPO model
    logger.info("Creating PPO model...")
    logger.info(f"  Learning rate: {config.LEARNING_RATE}")
    logger.info(f"  N steps: {config.N_STEPS}")
    logger.info(f"  Batch size: {config.BATCH_SIZE}")
    logger.info(f"  N epochs: {config.N_EPOCHS}")
    logger.info(f"  Gamma: {config.GAMMA}")
    logger.info(f"  Entropy coef: {config.ENT_COEF}")
    
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=config.LEARNING_RATE,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        n_epochs=config.N_EPOCHS,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        clip_range=config.CLIP_RANGE,
        ent_coef=config.ENT_COEF,
        vf_coef=config.VF_COEF,
        max_grad_norm=config.MAX_GRAD_NORM,
        tensorboard_log=config.TENSORBOARD_LOG,
        verbose=1,
        device='auto',  # Use GPU if available
    )
    
    logger.info(f"Model created: {model.policy}")
    logger.info("-"*80)
    
    # 6. Create callbacks
    logger.info("Setting up callbacks...")
    callbacks = create_callbacks(config, eval_env)
    logger.info(f"  Checkpoint frequency: {config.CHECKPOINT_FREQ:,} steps")
    logger.info(f"  Checkpoint directory: {config.CHECKPOINT_DIR}")
    logger.info(f"  TensorBoard log: {config.TENSORBOARD_LOG}")
    logger.info(f"  Early stopping patience: {config.EARLY_STOP_PATIENCE} evaluations")
    logger.info(f"  Early stopping min delta: {config.EARLY_STOP_MIN_DELTA}")
    logger.info("-"*80)
    
    # 7. Start training
    logger.info("="*80)
    logger.info(f"STARTING TRAINING: {config.TOTAL_TIMESTEPS:,} timesteps")
    logger.info("="*80)
    
    start_time = datetime.now()
    
    try:
        model.learn(
            total_timesteps=config.TOTAL_TIMESTEPS,
            callback=callbacks,
            log_interval=10,  # Log every 10 updates
            progress_bar=True,
        )
        
        training_time = datetime.now() - start_time
        logger.info("="*80)
        logger.info("TRAINING COMPLETED!")
        logger.info(f"Training time: {training_time}")
        logger.info("="*80)
        
        # 8. Save final model
        final_model_path = f"{config.CHECKPOINT_DIR}/xauusd_ppo_final"
        model.save(final_model_path)
        train_env.save(f"{final_model_path}_vecnormalize.pkl")
        logger.info(f"Final model saved: {final_model_path}")
        
        # 9. Show training summary
        logger.info("\n" + "="*80)
        logger.info("TRAINING SUMMARY")
        logger.info("="*80)
        logger.info(f"Total timesteps: {config.TOTAL_TIMESTEPS:,}")
        logger.info(f"Training time: {training_time}")
        logger.info(f"Checkpoints saved: {config.CHECKPOINT_DIR}")
        logger.info(f"TensorBoard logs: {config.TENSORBOARD_LOG}")
        logger.info("")
        logger.info(f"Train set: {len(train_df):,} rows")
        logger.info(f"Validation set: {len(val_df):,} rows")
        logger.info(f"Hidden test set: {len(test_df):,} rows (NOT used in training)")
        logger.info("")
        logger.info("⚠️  IMPORTANT: Test set (2025-2026) was NEVER seen by the model!")
        logger.info("Use test_model.py to evaluate on this hidden test set.")
        logger.info("\nTo view training progress:")
        logger.info(f"  tensorboard --logdir {config.TENSORBOARD_LOG}")
        logger.info("="*80)
        
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user!")
        logger.info("Saving current model...")
        interrupted_model_path = f"{config.CHECKPOINT_DIR}/xauusd_ppo_interrupted"
        model.save(interrupted_model_path)
        train_env.save(f"{interrupted_model_path}_vecnormalize.pkl")
        logger.info(f"Model saved: {interrupted_model_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Close environments
        train_env.close()
        eval_env.close()


def main():
    """Main execution"""
    try:
        config = TrainingConfig()
        
        # Check if data file exists
        if not Path(config.DATA_FILE).exists():
            logger.error(f"Data file not found: {config.DATA_FILE}")
            logger.error("Please run prepare_for_training.py first!")
            sys.exit(1)
        
        # Start training
        train_model(config)
        
        logger.info("\n✅ Training pipeline completed successfully!")
        
    except Exception as e:
        logger.critical(f"Critical error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

