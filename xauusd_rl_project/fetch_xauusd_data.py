"""
XAUUSD Historical Data Fetcher from MetaTrader 5
=================================================
This script fetches historical OHLCV data for XAUUSD from MT5 Terminal
and saves it in Parquet format for AI training purposes.

Author: Kasidit Sangsipet
Account: 2000730
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Optional, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xauusd_data_fetch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MT5DataFetcher:
    """Class to handle MetaTrader 5 data fetching operations"""
    
    SYMBOL = "XAUUSD"
    TIMEFRAMES = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'H1': mt5.TIMEFRAME_H1
    }
    
    def __init__(self, data_dir: str = "data", years_back: int = 5):
        """
        Initialize the MT5 Data Fetcher
        
        Args:
            data_dir: Directory to save the parquet files
            years_back: Number of years of historical data to fetch (default: 5)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.connected = False
        self.years_back = years_back
        
    def connect(self) -> bool:
        """
        Establish connection to MT5 Terminal
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Initialize MT5 connection
            if not mt5.initialize():
                logger.error(f"MT5 initialization failed, error code: {mt5.last_error()}")
                return False
            
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account info")
                return False
            
            logger.info(f"Connected to MT5 Terminal")
            logger.info(f"Account: {account_info.login}: {account_info.name}")
            logger.info(f"Server: {account_info.server}")
            logger.info(f"Balance: {account_info.balance} {account_info.currency}")
            
            # Verify symbol availability
            symbol_info = mt5.symbol_info(self.SYMBOL)
            if symbol_info is None:
                logger.error(f"Symbol {self.SYMBOL} not found")
                return False
            
            if not symbol_info.visible:
                logger.info(f"Symbol {self.SYMBOL} is not visible, attempting to enable...")
                if not mt5.symbol_select(self.SYMBOL, True):
                    logger.error(f"Failed to enable symbol {self.SYMBOL}")
                    return False
            
            logger.info(f"Symbol {self.SYMBOL} is ready")
            self.connected = True
            return True
            
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            return False
    
    def fetch_data(self, timeframe_key: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a specific timeframe
        
        Args:
            timeframe_key: Timeframe key (e.g., 'M1', 'M5', 'M15', 'H1')
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if not self.connected:
            logger.error("Not connected to MT5. Call connect() first.")
            return None
        
        try:
            timeframe = self.TIMEFRAMES[timeframe_key]
            
            # Calculate date range - from N years ago to now
            date_to = datetime.now()
            date_from = date_to - timedelta(days=365 * self.years_back)
            
            logger.info(f"Fetching {timeframe_key} data for {self.SYMBOL}...")
            logger.info(f"  Requested range: {date_from.strftime('%Y-%m-%d')} to {date_to.strftime('%Y-%m-%d')} ({self.years_back} years)")
            
            # Fetch rates using date range (gets all available data in the range)
            rates = mt5.copy_rates_range(
                self.SYMBOL,
                timeframe,
                date_from,
                date_to
            )
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No data available for {timeframe_key}, error: {mt5.last_error()}")
                logger.info(f"  Attempting to fetch whatever data is available...")
                
                # Fallback: try to get at least some data from current position
                rates = mt5.copy_rates_from_pos(self.SYMBOL, timeframe, 0, 10000)
                
                if rates is None or len(rates) == 0:
                    logger.error(f"Failed to fetch any {timeframe_key} data")
                    return None
                else:
                    logger.info(f"  Fetched {len(rates):,} bars using fallback method")
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            
            # Convert Unix timestamp to datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Select and order important columns
            columns_to_keep = ['time', 'open', 'high', 'low', 'close', 
                             'tick_volume', 'spread', 'real_volume']
            df = df[columns_to_keep]
            
            # Calculate actual time span
            actual_days = (df['time'].max() - df['time'].min()).days
            actual_years = actual_days / 365.25
            
            # Log statistics
            logger.info(f"✓ {timeframe_key}: Fetched {len(df):,} bars")
            logger.info(f"  Actual range: {df['time'].min()} to {df['time'].max()}")
            logger.info(f"  Time span: {actual_days:,} days ({actual_years:.2f} years)")
            logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            return df
            
        except KeyError:
            logger.error(f"Invalid timeframe key: {timeframe_key}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {timeframe_key} data: {str(e)}")
            return None
    
    def save_to_parquet(self, df: pd.DataFrame, timeframe_key: str) -> bool:
        """
        Save DataFrame to Parquet format
        
        Args:
            df: DataFrame to save
            timeframe_key: Timeframe identifier for filename
            
        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            filename = self.data_dir / f"{self.SYMBOL}_{timeframe_key}.parquet"
            
            # Save with compression for optimal storage
            df.to_parquet(
                filename,
                engine='pyarrow',
                compression='snappy',
                index=False
            )
            
            file_size = filename.stat().st_size / 1024 / 1024
            logger.info(f"✓ Saved to {filename} ({file_size:.2f} MB)")
            return True
            
        except Exception as e:
            logger.error(f"Error saving {timeframe_key} to parquet: {str(e)}")
            return False
    
    def fetch_all_timeframes(self) -> Dict[str, bool]:
        """
        Fetch and save data for all configured timeframes
        
        Returns:
            Dictionary with timeframe keys and success status
        """
        results = {}
        
        logger.info("="*60)
        logger.info("Starting XAUUSD Historical Data Download")
        logger.info(f"Target: {self.years_back} years of historical data")
        logger.info("="*60)
        
        for timeframe_key in self.TIMEFRAMES.keys():
            try:
                df = self.fetch_data(timeframe_key)
                
                if df is not None and not df.empty:
                    success = self.save_to_parquet(df, timeframe_key)
                    results[timeframe_key] = success
                else:
                    results[timeframe_key] = False
                    
                logger.info("-" * 60)
                
            except Exception as e:
                logger.error(f"Unexpected error processing {timeframe_key}: {str(e)}")
                results[timeframe_key] = False
        
        return results
    
    def disconnect(self):
        """Shutdown MT5 connection properly"""
        if self.connected:
            mt5.shutdown()
            logger.info("MT5 connection closed")
            self.connected = False
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures connection is always closed"""
        self.disconnect()


def main():
    """Main execution function"""
    try:
        # Use context manager to ensure proper cleanup
        # You can change years_back parameter to fetch more or less historical data
        # Default is 5 years, but will fetch as much as available
        with MT5DataFetcher(data_dir="data", years_back=5) as fetcher:
            if not fetcher.connected:
                logger.error("Failed to connect to MT5. Exiting.")
                return
            
            # Fetch all timeframes
            results = fetcher.fetch_all_timeframes()
            
            # Summary report
            logger.info("="*60)
            logger.info("DOWNLOAD SUMMARY")
            logger.info("="*60)
            
            success_count = sum(results.values())
            total_count = len(results)
            
            for timeframe, success in results.items():
                status = "✓ SUCCESS" if success else "✗ FAILED"
                logger.info(f"{timeframe:6s}: {status}")
            
            logger.info("-" * 60)
            logger.info(f"Completed: {success_count}/{total_count} timeframes")
            logger.info("="*60)
            
            if success_count == total_count:
                logger.info("All data fetched successfully! Ready for AI training.")
            else:
                logger.warning("Some timeframes failed. Please check the logs.")
    
    except Exception as e:
        logger.critical(f"Critical error in main execution: {str(e)}")
        raise
    finally:
        logger.info("Script execution completed")


if __name__ == "__main__":
    main()

