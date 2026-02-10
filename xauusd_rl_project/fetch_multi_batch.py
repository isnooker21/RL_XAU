"""
Multi-Batch Historical Data Fetcher for Short Timeframes
=========================================================
This script fetches historical data in multiple batches to overcome
the ~100,000 bars limit of MT5 API for M1 and M5 timeframes.

Usage: python fetch_multi_batch.py
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys

# Configure logging
file_handler = logging.FileHandler('xauusd_multi_batch.log', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
logger = logging.getLogger(__name__)

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class MultiBatchFetcher:
    """Fetch data in multiple batches to overcome MT5 limits"""
    
    SYMBOL = "XAUUSD.s"  # Change if your broker uses different symbol
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def connect(self):
        """Connect to MT5"""
        if not mt5.initialize():
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        
        account_info = mt5.account_info()
        if account_info:
            logger.info(f"Connected to MT5: {account_info.login} - {account_info.name}")
        
        # Enable symbol
        if not mt5.symbol_select(self.SYMBOL, True):
            logger.warning(f"Could not enable {self.SYMBOL}, trying without suffix...")
            self.SYMBOL = "XAUUSD"
            if not mt5.symbol_select(self.SYMBOL, True):
                logger.error("Could not enable XAUUSD symbol")
                return False
        
        logger.info(f"Symbol {self.SYMBOL} ready")
        return True
    
    def fetch_multi_batch(self, timeframe, timeframe_name, years_back=5, batch_days=90):
        """
        Fetch data in multiple batches
        
        Args:
            timeframe: MT5 timeframe constant
            timeframe_name: Name for logging (e.g., 'M1', 'M5')
            years_back: Total years to fetch
            batch_days: Days per batch (smaller = more batches but complete data)
        """
        logger.info(f"[{timeframe_name}] Starting multi-batch download...")
        
        all_data = []
        date_to = datetime.now()
        date_from = date_to - timedelta(days=365 * years_back)
        
        current_date = date_to
        batch_num = 1
        total_bars = 0
        
        while current_date > date_from:
            batch_start = current_date - timedelta(days=batch_days)
            if batch_start < date_from:
                batch_start = date_from
            
            logger.info(f"[{timeframe_name}] Batch {batch_num}: {batch_start.strftime('%Y-%m-%d')} to {current_date.strftime('%Y-%m-%d')}")
            
            rates = mt5.copy_rates_range(self.SYMBOL, timeframe, batch_start, current_date)
            
            if rates is not None and len(rates) > 0:
                df_batch = pd.DataFrame(rates)
                all_data.append(df_batch)
                total_bars += len(df_batch)
                logger.info(f"[{timeframe_name}] Batch {batch_num}: Fetched {len(df_batch):,} bars (Total: {total_bars:,})")
            else:
                logger.warning(f"[{timeframe_name}] Batch {batch_num}: No data")
            
            current_date = batch_start
            batch_num += 1
        
        if not all_data:
            logger.error(f"[{timeframe_name}] No data fetched!")
            return None
        
        # Combine all batches
        logger.info(f"[{timeframe_name}] Combining {len(all_data)} batches...")
        df = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates and sort
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)
        
        # Select columns
        columns_to_keep = ['time', 'open', 'high', 'low', 'close', 
                          'tick_volume', 'spread', 'real_volume']
        df = df[columns_to_keep]
        
        # Statistics
        actual_days = (df['time'].max() - df['time'].min()).days
        actual_years = actual_days / 365.25
        
        logger.info(f"[SUCCESS] {timeframe_name}: {len(df):,} unique bars")
        logger.info(f"  Range: {df['time'].min()} to {df['time'].max()}")
        logger.info(f"  Span: {actual_days:,} days ({actual_years:.2f} years)")
        
        return df
    
    def save_parquet(self, df, timeframe_name):
        """Save to Parquet"""
        filename = self.data_dir / f"{self.SYMBOL}_{timeframe_name}_FULL.parquet"
        df.to_parquet(filename, engine='pyarrow', compression='snappy', index=False)
        
        file_size = filename.stat().st_size / 1024 / 1024
        logger.info(f"[SAVED] {filename} ({file_size:.2f} MB)")
    
    def disconnect(self):
        """Disconnect from MT5"""
        mt5.shutdown()
        logger.info("MT5 disconnected")


def main():
    """Main execution"""
    fetcher = MultiBatchFetcher(data_dir="data")
    
    if not fetcher.connect():
        logger.error("Connection failed!")
        return
    
    try:
        logger.info("="*60)
        logger.info("Multi-Batch Data Download")
        logger.info("="*60)
        
        # Fetch M1 data (90-day batches to avoid limit)
        df_m1 = fetcher.fetch_multi_batch(
            mt5.TIMEFRAME_M1, 
            "M1", 
            years_back=5, 
            batch_days=90  # 90 days = ~90,000 bars (under limit)
        )
        if df_m1 is not None:
            fetcher.save_parquet(df_m1, "M1")
        
        logger.info("-"*60)
        
        # Fetch M5 data (180-day batches)
        df_m5 = fetcher.fetch_multi_batch(
            mt5.TIMEFRAME_M5, 
            "M5", 
            years_back=5, 
            batch_days=180  # 180 days = ~50,000 bars
        )
        if df_m5 is not None:
            fetcher.save_parquet(df_m5, "M5")
        
        logger.info("="*60)
        logger.info("Multi-batch download completed!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        fetcher.disconnect()


if __name__ == "__main__":
    main()

