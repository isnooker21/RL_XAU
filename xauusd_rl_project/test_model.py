"""
Test Trained RL Model
======================
Load and test trained PPO model on test data

Usage:
    python test_model.py --model models/xauusd_model_checkpoints/xauusd_ppo_final.zip
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from trading_env import TradingEnv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_model(model_path: str, vecnorm_path: str = None):
    """Load trained model and vec normalize"""
    logger.info(f"Loading model from: {model_path}")
    
    model = PPO.load(model_path)
    
    # Load VecNormalize if available
    if vecnorm_path and Path(vecnorm_path).exists():
        logger.info(f"Loading VecNormalize from: {vecnorm_path}")
    
    return model


def test_model(model_path: str, 
               data_file: str = "../processed_data/XAUUSD_READY_TO_TRAIN.parquet",
               n_episodes: int = 10):
    """
    Test trained model
    
    Args:
        model_path: Path to trained model
        data_file: Path to test data
        n_episodes: Number of test episodes
    """
    logger.info("="*80)
    logger.info("TESTING RL MODEL")
    logger.info("="*80)
    
    # Load data
    df = pd.read_parquet(data_file)
    
    # Use last 20% as test set
    test_split = int(len(df) * 0.8)
    test_df = df.iloc[test_split:].reset_index(drop=True)
    logger.info(f"Test data: {len(test_df):,} rows")
    
    # Create environment
    def make_env():
        env = TradingEnv(
            df=test_df,
            initial_balance=10000.0,
            lot_size=0.01,
            commission=0.0,
        )
        return Monitor(env)
    
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize if exists
    vecnorm_path = model_path.replace('.zip', '_vecnormalize.pkl')
    if Path(vecnorm_path).exists():
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False  # Don't update normalization stats
        env.norm_reward = False
    
    # Load model
    model = PPO.load(model_path)
    logger.info("Model loaded successfully")
    logger.info("-"*80)
    
    # Test episodes
    results = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            episode_steps += 1
            
            if done:
                break
        
        # Get final info
        final_info = info[0]
        
        results.append({
            'episode': episode + 1,
            'steps': episode_steps,
            'reward': episode_reward,
            'final_balance': final_info['balance'],
            'final_equity': final_info['equity'],
            'total_profit': final_info['total_profit'],
            'total_trades': final_info['total_trades'],
            'win_rate': final_info['win_rate'],
        })
        
        logger.info(f"Episode {episode+1}/{n_episodes}:")
        logger.info(f"  Steps: {episode_steps}")
        logger.info(f"  Reward: {episode_reward:.4f}")
        logger.info(f"  Final Balance: ${final_info['balance']:.2f}")
        logger.info(f"  Profit: ${final_info['total_profit']:.2f}")
        logger.info(f"  Trades: {final_info['total_trades']} (Win Rate: {final_info['win_rate']:.1%})")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    results_df = pd.DataFrame(results)
    
    logger.info(f"Average Reward: {results_df['reward'].mean():.4f} ± {results_df['reward'].std():.4f}")
    logger.info(f"Average Profit: ${results_df['total_profit'].mean():.2f} ± ${results_df['total_profit'].std():.2f}")
    logger.info(f"Average Trades: {results_df['total_trades'].mean():.1f}")
    logger.info(f"Average Win Rate: {results_df['win_rate'].mean():.1%}")
    logger.info(f"Best Episode: ${results_df['total_profit'].max():.2f}")
    logger.info(f"Worst Episode: ${results_df['total_profit'].min():.2f}")
    
    # Plot results
    plot_results(results_df)
    
    env.close()
    
    return results_df


def plot_results(results_df: pd.DataFrame):
    """Plot test results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Profit per episode
    axes[0, 0].bar(results_df['episode'], results_df['total_profit'])
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Profit ($)')
    axes[0, 0].set_title('Profit per Episode')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Win rate
    axes[0, 1].bar(results_df['episode'], results_df['win_rate'] * 100)
    axes[0, 1].axhline(y=50, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Win Rate (%)')
    axes[0, 1].set_title('Win Rate per Episode')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Number of trades
    axes[1, 0].bar(results_df['episode'], results_df['total_trades'])
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Number of Trades')
    axes[1, 0].set_title('Trades per Episode')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Reward
    axes[1, 1].plot(results_df['episode'], results_df['reward'], marker='o')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Episode Reward')
    axes[1, 1].set_title('Reward per Episode')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"\nPlot saved: {output_file}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test trained RL model")
    parser.add_argument(
        '--model',
        type=str,
        default='models/xauusd_model_checkpoints/xauusd_ppo_final.zip',
        help='Path to trained model'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='../processed_data/XAUUSD_READY_TO_TRAIN.parquet',
        help='Path to test data'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Number of test episodes'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        logger.error("Please train the model first using train_rl_model.py")
        sys.exit(1)
    
    # Run test
    try:
        results = test_model(args.model, args.data, args.episodes)
        logger.info("\n✅ Testing completed successfully!")
        
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

