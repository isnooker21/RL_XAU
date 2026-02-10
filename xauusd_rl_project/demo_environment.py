"""
Demo Trading Environment
========================
ทดสอบว่า Environment ทำงานได้ถูกต้องก่อนเริ่มฝึกจริง

Usage: python demo_environment.py
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from trading_env import TradingEnv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def demo_random_actions():
    """ทดสอบ Environment ด้วย random actions"""
    
    logger.info("="*80)
    logger.info("DEMO: Testing Trading Environment with Random Actions")
    logger.info("="*80)
    
    # 1. Load data
    data_file = "../processed_data/XAUUSD_READY_TO_TRAIN.parquet"
    
    if not Path(data_file).exists():
        logger.error(f"Data file not found: {data_file}")
        logger.error("Please run prepare_for_training.py first!")
        return
    
    logger.info(f"Loading data from: {data_file}")
    df = pd.read_parquet(data_file)
    logger.info(f"Data loaded: {len(df):,} rows")
    logger.info("-"*80)
    
    # 2. Create environment
    logger.info("Creating trading environment...")
    env = TradingEnv(
        df=df,
        initial_balance=10000.0,
        lot_size=0.01,
        commission=0.0,
    )
    logger.info(f"Environment created:")
    logger.info(f"  Observation space: {env.observation_space.shape}")
    logger.info(f"  Action space: {env.action_space.n} actions")
    logger.info("-"*80)
    
    # 3. Test episodes with random actions
    logger.info("Running test episodes with random actions...")
    logger.info("")
    
    n_episodes = 3
    max_steps = 200  # จำกัด steps เพื่อให้ demo เร็ว
    
    episode_results = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        logger.info(f"Episode {episode + 1}/{n_episodes}:")
        logger.info(f"  Starting balance: ${info['balance']:.2f}")
        
        for step in range(max_steps):
            # Random action
            action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            # Log every 50 steps
            if step % 50 == 0 and step > 0:
                action_names = ['Hold', 'Buy', 'Sell', 'Close']
                logger.info(f"    Step {step}: Action={action_names[action]}, "
                           f"Balance=${info['balance']:.2f}, "
                           f"Position={info['position']}, "
                           f"Reward={reward:.4f}")
            
            if terminated or truncated:
                break
        
        # Episode summary
        logger.info(f"  Episode finished:")
        logger.info(f"    Steps: {episode_steps}")
        logger.info(f"    Total Reward: {episode_reward:.4f}")
        logger.info(f"    Final Balance: ${info['balance']:.2f}")
        logger.info(f"    Final Equity: ${info['equity']:.2f}")
        logger.info(f"    Total Profit: ${info['total_profit']:.2f}")
        logger.info(f"    Total Trades: {info['total_trades']}")
        logger.info(f"    Win Rate: {info['win_rate']:.1%}")
        logger.info("")
        
        episode_results.append({
            'episode': episode + 1,
            'steps': episode_steps,
            'reward': episode_reward,
            'profit': info['total_profit'],
            'trades': info['total_trades'],
            'win_rate': info['win_rate'],
        })
    
    # Overall summary
    logger.info("-"*80)
    logger.info("SUMMARY OF RANDOM ACTIONS TEST")
    logger.info("-"*80)
    
    results_df = pd.DataFrame(episode_results)
    
    logger.info(f"Average Reward: {results_df['reward'].mean():.4f}")
    logger.info(f"Average Profit: ${results_df['profit'].mean():.2f}")
    logger.info(f"Average Trades: {results_df['trades'].mean():.1f}")
    logger.info(f"Average Win Rate: {results_df['win_rate'].mean():.1%}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ Environment test completed successfully!")
    logger.info("="*80)
    logger.info("\nNext steps:")
    logger.info("  1. Start training: python train_rl_model.py")
    logger.info("  2. Monitor progress: ./view_tensorboard.sh")
    logger.info("="*80)


def demo_specific_actions():
    """ทดสอบด้วย actions เฉพาะ"""
    
    logger.info("\n" + "="*80)
    logger.info("DEMO: Testing Specific Trading Scenarios")
    logger.info("="*80)
    
    # Load data
    data_file = "../processed_data/XAUUSD_READY_TO_TRAIN.parquet"
    df = pd.read_parquet(data_file)
    
    # Create environment
    env = TradingEnv(df=df, initial_balance=10000.0, lot_size=0.01)
    obs, info = env.reset()
    
    logger.info("Scenario 1: Buy -> Hold -> Close")
    logger.info("-"*40)
    
    # Action 1: Buy
    obs, reward, terminated, truncated, info = env.step(1)  # Buy
    logger.info(f"1. BUY  - Position: {info['position']}, Balance: ${info['balance']:.2f}, Reward: {reward:.4f}")
    
    # Action 2-5: Hold
    for i in range(5):
        obs, reward, terminated, truncated, info = env.step(0)  # Hold
        logger.info(f"{i+2}. HOLD - Position: {info['position']}, "
                   f"Equity: ${info['equity']:.2f}, "
                   f"Position Profit: ${info['position_profit']:.2f}, "
                   f"Reward: {reward:.4f}")
    
    # Action 6: Close
    obs, reward, terminated, truncated, info = env.step(3)  # Close
    logger.info(f"7. CLOSE - Position: {info['position']}, "
               f"Balance: ${info['balance']:.2f}, "
               f"Total Profit: ${info['total_profit']:.2f}, "
               f"Reward: {reward:.4f}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ Specific scenario test completed!")
    logger.info("="*80)


def main():
    """Main execution"""
    try:
        # Test 1: Random actions
        demo_random_actions()
        
        # Test 2: Specific actions
        # demo_specific_actions()  # Uncomment to test
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

