"""
Multi-Timeframe XAUUSD Data Fetcher with Alignment
===================================================
Fetch H4, H1, and M15 data with timestamp alignment for AI training.

Strategy:
- H4 & H1: Target 10+ years for trend analysis
- M15: Target 5+ years for entry points
- All timeframes aligned on common timestamps

Author: Kasidit Sangsipet
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
import sys

# Configure logging with UTF-8 encoding for Windows compatibility
file_handler = logging.FileHandler('xauusd_multi_tf.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
logger = logging.getLogger(__name__)

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class MultiTimeframeFetcher:
    """Fetch and align multiple timeframes for AI training"""
    
    SYMBOL = "XAUUSD"  # Will auto-detect suffix (.s, .i, etc.)
    
    # Target timeframes for multi-timeframe analysis
    TIMEFRAMES = {
        'H4': {'mt5': mt5.TIMEFRAME_H4, 'years': 10, 'desc': 'Trend Analysis'},
        'H1': {'mt5': mt5.TIMEFRAME_H1, 'years': 10, 'desc': 'Market Structure'},
        'M15': {'mt5': mt5.TIMEFRAME_M15, 'years': 5, 'desc': 'Entry Points'}
    }
    
    def __init__(self, raw_dir: str = "data", processed_dir: str = "processed_data"):
        """
        Initialize Multi-Timeframe Fetcher
        
        Args:
            raw_dir: Directory for raw parquet files
            processed_dir: Directory for aligned/processed data
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        self.connected = False
        self.actual_symbol = None
        
    def connect(self) -> bool:
        """Establish connection to MT5 Terminal"""
        try:
            if not mt5.initialize():
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account info")
                return False
            
            logger.info(f"Connected to MT5 Terminal")
            logger.info(f"Account: {account_info.login}: {account_info.name}")
            logger.info(f"Server: {account_info.server}")
            
            # Auto-detect symbol with suffix
            self.actual_symbol = self._detect_symbol()
            if not self.actual_symbol:
                logger.error("Could not find XAUUSD symbol")
                return False
            
            logger.info(f"Symbol detected: {self.actual_symbol}")
            self.connected = True
            return True
            
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            return False
    
    def _detect_symbol(self) -> Optional[str]:
        """Auto-detect XAUUSD symbol with various suffixes"""
        # Try common variations
        variations = [
            "XAUUSD",
            "XAUUSD.s",
            "XAUUSD.i",
            "XAUUSD.",
            "XAUUSDm",
            "GOLD",
            "GOLDm"
        ]
        
        for symbol in variations:
            if mt5.symbol_select(symbol, True):
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info and symbol_info.visible:
                    return symbol
        
        return None
    
    def fetch_timeframe(self, tf_key: str, batch_size_days: int = 365) -> Optional[pd.DataFrame]:
        """
        Fetch data for a specific timeframe using multi-batch approach
        
        Args:
            tf_key: Timeframe key ('H4', 'H1', 'M15')
            batch_size_days: Days per batch for large datasets
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if not self.connected:
            logger.error("Not connected to MT5")
            return None
        
        try:
            tf_config = self.TIMEFRAMES[tf_key]
            timeframe = tf_config['mt5']
            years_target = tf_config['years']
            
            logger.info(f"[{tf_key}] Fetching data ({tf_config['desc']})...")
            logger.info(f"[{tf_key}] Target: {years_target} years")
            
            # Calculate date range
            date_to = datetime.now()
            date_from = date_to - timedelta(days=365 * years_target)
            
            # Try single fetch first
            rates = mt5.copy_rates_range(
                self.actual_symbol,
                timeframe,
                date_from,
                date_to
            )
            
            # If single fetch fails or gets limited data, try multi-batch
            if rates is None or len(rates) == 0:
                logger.warning(f"[{tf_key}] Single fetch failed, trying multi-batch...")
                rates = self._fetch_multi_batch(timeframe, tf_key, date_from, date_to, batch_size_days)
            
            if rates is None or len(rates) == 0:
                logger.error(f"[{tf_key}] Failed to fetch any data")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Select and order columns
            columns_to_keep = ['time', 'open', 'high', 'low', 'close', 
                             'tick_volume', 'spread', 'real_volume']
            df = df[columns_to_keep]
            
            # Remove duplicates and sort
            df = df.drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)
            
            # Calculate statistics
            actual_days = (df['time'].max() - df['time'].min()).days
            actual_years = actual_days / 365.25
            
            logger.info(f"[SUCCESS] {tf_key}: {len(df):,} bars")
            logger.info(f"  Range: {df['time'].min()} to {df['time'].max()}")
            logger.info(f"  Span: {actual_days:,} days ({actual_years:.2f} years)")
            logger.info(f"  Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            return df
            
        except Exception as e:
            logger.error(f"[{tf_key}] Error fetching data: {str(e)}")
            return None
    
    def _fetch_multi_batch(self, timeframe, tf_key: str, date_from: datetime, 
                          date_to: datetime, batch_days: int) -> Optional[np.ndarray]:
        """Fetch data in multiple batches to overcome MT5 limits"""
        all_rates = []
        current_end = date_to
        batch_num = 1
        
        while current_end > date_from:
            current_start = current_end - timedelta(days=batch_days)
            if current_start < date_from:
                current_start = date_from
            
            logger.info(f"[{tf_key}] Batch {batch_num}: "
                       f"{current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
            
            rates = mt5.copy_rates_range(self.actual_symbol, timeframe, current_start, current_end)
            
            if rates is not None and len(rates) > 0:
                all_rates.append(rates)
                logger.info(f"[{tf_key}] Batch {batch_num}: {len(rates):,} bars")
            
            current_end = current_start
            batch_num += 1
            
            # Safety limit
            if batch_num > 50:
                logger.warning(f"[{tf_key}] Reached batch limit, stopping")
                break
        
        if not all_rates:
            return None
        
        # Combine all batches
        combined = np.concatenate(all_rates[::-1])  # Reverse to get chronological order
        logger.info(f"[{tf_key}] Combined {len(all_rates)} batches: {len(combined):,} total bars")
        
        return combined
    
    def save_raw_data(self, df: pd.DataFrame, tf_key: str) -> bool:
        """Save raw timeframe data to parquet"""
        try:
            filename = self.raw_dir / f"{self.actual_symbol}_{tf_key}.parquet"
            df.to_parquet(filename, engine='pyarrow', compression='snappy', index=False)
            
            file_size = filename.stat().st_size / 1024 / 1024
            logger.info(f"[SAVED] {filename} ({file_size:.2f} MB)")
            return True
            
        except Exception as e:
            logger.error(f"Error saving {tf_key}: {str(e)}")
            return False
    
    def align_timeframes(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Align all timeframes to common timestamps
        
        Strategy:
        - Use M15 as base (most granular)
        - Forward-fill H1 and H4 data to match M15 timestamps
        - This allows AI to see higher timeframe context for each M15 bar
        
        Args:
            dfs: Dictionary of DataFrames {tf_key: df}
            
        Returns:
            Dictionary of aligned DataFrames with synchronized timestamps
        """
        logger.info("="*60)
        logger.info("ALIGNING TIMEFRAMES")
        logger.info("="*60)
        
        # Find common date range (intersection of all timeframes)
        min_start = max(df['time'].min() for df in dfs.values())
        max_end = min(df['time'].max() for df in dfs.values())
        
        logger.info(f"Common date range: {min_start} to {max_end}")
        
        # Filter all dataframes to common range
        aligned_dfs = {}
        for tf_key, df in dfs.items():
            df_filtered = df[(df['time'] >= min_start) & (df['time'] <= max_end)].copy()
            aligned_dfs[tf_key] = df_filtered
            logger.info(f"[{tf_key}] After filtering: {len(df_filtered):,} bars")
        
        # Use M15 as base timeline
        if 'M15' not in aligned_dfs:
            logger.error("M15 timeframe missing, cannot align")
            return aligned_dfs
        
        base_timeline = aligned_dfs['M15']['time'].values
        logger.info(f"Using M15 as base: {len(base_timeline):,} timestamps")
        
        # Create aligned dataset for each timeframe
        for tf_key in ['H4', 'H1']:
            if tf_key not in aligned_dfs:
                continue
            
            df_tf = aligned_dfs[tf_key].copy()
            df_m15 = aligned_dfs['M15'].copy()
            
            # Merge with M15 timeline using forward fill
            # This ensures every M15 bar has corresponding H4/H1 data
            df_tf_aligned = pd.merge_asof(
                df_m15[['time']],  # M15 timestamps
                df_tf,  # Higher timeframe data
                on='time',
                direction='backward',  # Use most recent H4/H1 bar
                suffixes=('', f'_{tf_key}')
            )
            
            # Rename columns to include timeframe prefix
            rename_cols = {}
            for col in ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']:
                if col in df_tf_aligned.columns:
                    rename_cols[col] = f'{col}_{tf_key}'
            
            df_tf_aligned = df_tf_aligned.rename(columns=rename_cols)
            aligned_dfs[f'{tf_key}_aligned'] = df_tf_aligned
            
            logger.info(f"[{tf_key}] Aligned to M15: {len(df_tf_aligned):,} rows")
        
        return aligned_dfs
    
    def create_combined_dataset(self, aligned_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create single combined dataset with all timeframes
        
        Structure:
        time | open_M15 | high_M15 | ... | open_H1 | high_H1 | ... | open_H4 | high_H4 | ...
        
        This allows AI to see multi-timeframe context in one row
        """
        logger.info("Creating combined multi-timeframe dataset...")
        
        # Start with M15 as base
        combined = aligned_dfs['M15'].copy()
        combined = combined.rename(columns={
            col: f'{col}_M15' for col in combined.columns if col != 'time'
        })
        
        # Merge H1 aligned data
        if 'H1_aligned' in aligned_dfs:
            h1_data = aligned_dfs['H1_aligned'].copy()
            combined = pd.merge(combined, h1_data, on='time', how='left')
        
        # Merge H4 aligned data
        if 'H4_aligned' in aligned_dfs:
            h4_data = aligned_dfs['H4_aligned'].copy()
            combined = pd.merge(combined, h4_data, on='time', how='left')
        
        # Fill any remaining NaN values with forward fill
        combined = combined.fillna(method='ffill')
        
        logger.info(f"[COMBINED] Dataset created: {len(combined):,} rows, {len(combined.columns)} columns")
        logger.info(f"  Columns: {', '.join(combined.columns[:10])}...")
        
        return combined
    
    def save_processed_data(self, aligned_dfs: Dict[str, pd.DataFrame], 
                           combined: pd.DataFrame) -> bool:
        """Save aligned and combined datasets"""
        try:
            # Save aligned individual timeframes
            for tf_key in ['M15', 'H1', 'H4']:
                if tf_key in aligned_dfs:
                    filename = self.processed_dir / f"{self.actual_symbol}_{tf_key}_aligned.parquet"
                    aligned_dfs[tf_key].to_parquet(filename, engine='pyarrow', 
                                                   compression='snappy', index=False)
                    size = filename.stat().st_size / 1024 / 1024
                    logger.info(f"[SAVED] {filename} ({size:.2f} MB)")
            
            # Save combined multi-timeframe dataset
            combined_file = self.processed_dir / f"{self.actual_symbol}_COMBINED.parquet"
            combined.to_parquet(combined_file, engine='pyarrow', compression='snappy', index=False)
            size = combined_file.stat().st_size / 1024 / 1024
            logger.info(f"[SAVED] {combined_file} ({size:.2f} MB)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            return False
    
    def disconnect(self):
        """Shutdown MT5 connection"""
        if self.connected:
            mt5.shutdown()
            logger.info("MT5 connection closed")
            self.connected = False
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


def main():
    """Main execution function"""
    try:
        with MultiTimeframeFetcher(raw_dir="data", processed_dir="processed_data") as fetcher:
            if not fetcher.connected:
                logger.error("Failed to connect to MT5. Exiting.")
                return
            
            logger.info("="*60)
            logger.info("MULTI-TIMEFRAME DATA FETCHER")
            logger.info("Strategy: H4 (Trend) + H1 (Structure) + M15 (Entry)")
            logger.info("="*60)
            
            # Step 1: Fetch all timeframes
            raw_dfs = {}
            for tf_key in ['H4', 'H1', 'M15']:
                df = fetcher.fetch_timeframe(tf_key)
                if df is not None:
                    raw_dfs[tf_key] = df
                    fetcher.save_raw_data(df, tf_key)
                logger.info("-"*60)
            
            if len(raw_dfs) < 3:
                logger.error("Failed to fetch all required timeframes")
                return
            
            # Step 2: Align timeframes
            aligned_dfs = fetcher.align_timeframes(raw_dfs)
            
            # Step 3: Create combined dataset
            combined = fetcher.create_combined_dataset(aligned_dfs)
            
            # Step 4: Save processed data
            logger.info("-"*60)
            fetcher.save_processed_data(aligned_dfs, combined)
            
            # Final summary
            logger.info("="*60)
            logger.info("SUMMARY")
            logger.info("="*60)
            logger.info("RAW DATA (data/):")
            for tf_key, df in raw_dfs.items():
                days = (df['time'].max() - df['time'].min()).days
                years = days / 365.25
                logger.info(f"  {tf_key:4s}: {len(df):,} bars, {years:.2f} years")
            
            logger.info("\nPROCESSED DATA (processed_data/):")
            logger.info(f"  M15 (aligned): {len(aligned_dfs['M15']):,} bars")
            logger.info(f"  H1  (aligned): {len(aligned_dfs['H1']):,} bars")
            logger.info(f"  H4  (aligned): {len(aligned_dfs['H4']):,} bars")
            logger.info(f"  COMBINED: {len(combined):,} rows x {len(combined.columns)} columns")
            
            logger.info("="*60)
            logger.info("Multi-timeframe dataset ready for AI training!")
            logger.info("="*60)
    
    except Exception as e:
        logger.critical(f"Critical error: {str(e)}")
        raise
    finally:
        logger.info("Script execution completed")


if __name__ == "__main__":
    main()

