"""
Feature Engineering for RL Training
====================================
Prepare XAUUSD_COMBINED.parquet for Reinforcement Learning by adding:
- Log Returns
- Technical Indicators (RSI, ATR, MACD)
- Data Cleaning (NaN, Outliers)

Output: XAUUSD_READY_TO_TRAIN.parquet

Usage: python prepare_for_training.py
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Try to import pandas_ta, fallback to ta
try:
    import pandas_ta as ta
    USE_PANDAS_TA = True
    print("Using pandas_ta library")
except ImportError:
    try:
        import ta as ta_lib
        USE_PANDAS_TA = False
        print("Using ta library (fallback)")
    except ImportError:
        print("ERROR: Please install pandas_ta or ta library")
        print("Run: pip install pandas_ta")
        sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_engineering.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class FeatureEngineer:
    """Feature Engineering for Multi-Timeframe Trading Data"""
    
    TIMEFRAMES = ['M15', 'H1', 'H4']
    
    def __init__(self, input_file: str = "processed_data/XAUUSD_COMBINED.parquet",
                 output_file: str = "processed_data/XAUUSD_READY_TO_TRAIN.parquet"):
        """
        Initialize Feature Engineer
        
        Args:
            input_file: Path to COMBINED.parquet
            output_file: Path to save READY_TO_TRAIN.parquet
        """
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.df = None
        
        # Auto-detect input file if not found
        if not self.input_file.exists():
            parent = self.input_file.parent
            pattern = "*_COMBINED.parquet"
            files = list(parent.glob(pattern))
            if files:
                self.input_file = files[0]
                logger.info(f"Auto-detected input file: {self.input_file}")
            else:
                logger.error(f"Input file not found: {input_file}")
                logger.error("Please run fetch_data.py first!")
                sys.exit(1)
    
    def load_data(self) -> pd.DataFrame:
        """Load COMBINED dataset"""
        logger.info(f"Loading data from: {self.input_file}")
        
        self.df = pd.read_parquet(self.input_file)
        
        logger.info(f"Data loaded: {len(self.df):,} rows, {len(self.df.columns)} columns")
        logger.info(f"Date range: {self.df['time'].min()} to {self.df['time'].max()}")
        logger.info(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        return self.df
    
    def calculate_log_returns(self) -> None:
        """Calculate Log Returns for all timeframes"""
        logger.info("Calculating Log Returns...")
        
        for tf in self.TIMEFRAMES:
            close_col = f'close_{tf}'
            if close_col in self.df.columns:
                # Log Return = ln(price_t / price_t-1)
                self.df[f'log_return_{tf}'] = np.log(
                    self.df[close_col] / self.df[close_col].shift(1)
                )
                logger.info(f"  {tf}: log_return_{tf}")
            else:
                logger.warning(f"  {tf}: {close_col} not found, skipping")
    
    def calculate_rsi(self, period: int = 14) -> None:
        """
        Calculate RSI for all timeframes
        
        Args:
            period: RSI period (default: 14)
        """
        logger.info(f"Calculating RSI ({period})...")
        
        for tf in self.TIMEFRAMES:
            close_col = f'close_{tf}'
            if close_col not in self.df.columns:
                logger.warning(f"  {tf}: {close_col} not found, skipping")
                continue
            
            if USE_PANDAS_TA:
                # Using pandas_ta
                self.df[f'rsi_{tf}'] = ta.rsi(self.df[close_col], length=period)
            else:
                # Using ta library
                self.df[f'rsi_{tf}'] = ta_lib.momentum.RSIIndicator(
                    close=self.df[close_col],
                    window=period
                ).rsi()
            
            logger.info(f"  {tf}: rsi_{tf}")
    
    def calculate_atr(self, period: int = 14) -> None:
        """
        Calculate ATR for M15 and H1 (volatility measure)
        
        Args:
            period: ATR period (default: 14)
        """
        logger.info(f"Calculating ATR ({period}) for M15 and H1...")
        
        for tf in ['M15', 'H1']:  # Only M15 and H1
            high_col = f'high_{tf}'
            low_col = f'low_{tf}'
            close_col = f'close_{tf}'
            
            if not all(col in self.df.columns for col in [high_col, low_col, close_col]):
                logger.warning(f"  {tf}: Required columns not found, skipping")
                continue
            
            if USE_PANDAS_TA:
                # Using pandas_ta
                self.df[f'atr_{tf}'] = ta.atr(
                    high=self.df[high_col],
                    low=self.df[low_col],
                    close=self.df[close_col],
                    length=period
                )
            else:
                # Using ta library
                self.df[f'atr_{tf}'] = ta_lib.volatility.AverageTrueRange(
                    high=self.df[high_col],
                    low=self.df[low_col],
                    close=self.df[close_col],
                    window=period
                ).average_true_range()
            
            logger.info(f"  {tf}: atr_{tf}")
    
    def calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        """
        Calculate MACD for all timeframes
        
        Args:
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line period (default: 9)
        """
        logger.info(f"Calculating MACD ({fast}, {slow}, {signal})...")
        
        for tf in self.TIMEFRAMES:
            close_col = f'close_{tf}'
            if close_col not in self.df.columns:
                logger.warning(f"  {tf}: {close_col} not found, skipping")
                continue
            
            if USE_PANDAS_TA:
                # Using pandas_ta
                macd = ta.macd(
                    self.df[close_col],
                    fast=fast,
                    slow=slow,
                    signal=signal
                )
                self.df[f'macd_{tf}'] = macd[f'MACD_{fast}_{slow}_{signal}']
                self.df[f'macd_signal_{tf}'] = macd[f'MACDs_{fast}_{slow}_{signal}']
                self.df[f'macd_hist_{tf}'] = macd[f'MACDh_{fast}_{slow}_{signal}']
            else:
                # Using ta library
                macd_indicator = ta_lib.trend.MACD(
                    close=self.df[close_col],
                    window_fast=fast,
                    window_slow=slow,
                    window_sign=signal
                )
                self.df[f'macd_{tf}'] = macd_indicator.macd()
                self.df[f'macd_signal_{tf}'] = macd_indicator.macd_signal()
                self.df[f'macd_hist_{tf}'] = macd_indicator.macd_diff()
            
            logger.info(f"  {tf}: macd_{tf}, macd_signal_{tf}, macd_hist_{tf}")
    
    def add_additional_features(self) -> None:
        """Add useful features for RL"""
        logger.info("Adding additional features...")
        
        for tf in self.TIMEFRAMES:
            close_col = f'close_{tf}'
            high_col = f'high_{tf}'
            low_col = f'low_{tf}'
            open_col = f'open_{tf}'
            
            if not all(col in self.df.columns for col in [close_col, high_col, low_col, open_col]):
                continue
            
            # Price change (%)
            self.df[f'price_change_{tf}'] = (
                (self.df[close_col] - self.df[close_col].shift(1)) / 
                self.df[close_col].shift(1) * 100
            )
            
            # Candle body ratio
            self.df[f'body_ratio_{tf}'] = (
                abs(self.df[close_col] - self.df[open_col]) /
                (self.df[high_col] - self.df[low_col] + 1e-10)  # Avoid division by zero
            )
            
            # Moving Averages
            self.df[f'ma20_{tf}'] = self.df[close_col].rolling(20).mean()
            self.df[f'ma50_{tf}'] = self.df[close_col].rolling(50).mean()
            
            # Distance from MA (normalized)
            self.df[f'dist_ma20_{tf}'] = (
                (self.df[close_col] - self.df[f'ma20_{tf}']) / 
                self.df[f'ma20_{tf}'] * 100
            )
        
        logger.info("  Added: price_change, body_ratio, ma20, ma50, dist_ma20")
    
    def handle_nan_values(self) -> None:
        """Handle NaN values from indicator calculations"""
        logger.info("Handling NaN values...")
        
        # Count NaN before
        nan_count_before = self.df.isnull().sum().sum()
        logger.info(f"  NaN values before: {nan_count_before:,}")
        
        # Drop rows with NaN (from initial indicators like RSI, MACD)
        # Keep track of original length
        original_len = len(self.df)
        self.df = self.df.dropna()
        dropped_rows = original_len - len(self.df)
        
        logger.info(f"  Dropped {dropped_rows:,} rows with NaN")
        logger.info(f"  Remaining rows: {len(self.df):,}")
        
        # Verify no NaN left
        nan_count_after = self.df.isnull().sum().sum()
        if nan_count_after > 0:
            logger.warning(f"  NaN values still present: {nan_count_after}")
        else:
            logger.info("  All NaN values handled")
    
    def handle_outliers(self, n_std: float = 5.0) -> None:
        """
        Detect and clip outliers using z-score method
        
        Args:
            n_std: Number of standard deviations for outlier threshold
        """
        logger.info(f"Handling outliers (threshold: {n_std} std)...")
        
        outlier_cols = []
        
        # Check log returns and indicator columns
        for col in self.df.columns:
            if any(x in col for x in ['log_return', 'rsi', 'macd', 'price_change']):
                outlier_cols.append(col)
        
        total_clipped = 0
        
        for col in outlier_cols:
            if col not in self.df.columns:
                continue
            
            # Calculate z-score
            mean = self.df[col].mean()
            std = self.df[col].std()
            
            if std == 0:
                continue
            
            z_scores = np.abs((self.df[col] - mean) / std)
            outliers = z_scores > n_std
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                # Clip outliers
                lower_bound = mean - n_std * std
                upper_bound = mean + n_std * std
                self.df[col] = self.df[col].clip(lower_bound, upper_bound)
                total_clipped += outlier_count
                logger.info(f"  {col}: clipped {outlier_count} outliers")
        
        logger.info(f"  Total outliers clipped: {total_clipped:,}")
    
    def save_data(self) -> None:
        """Save engineered features to parquet"""
        logger.info(f"Saving to: {self.output_file}")
        
        # Ensure output directory exists
        self.output_file.parent.mkdir(exist_ok=True)
        
        # Save to parquet
        self.df.to_parquet(
            self.output_file,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        
        file_size = self.output_file.stat().st_size / 1024 / 1024
        logger.info(f"Saved: {self.output_file} ({file_size:.2f} MB)")
    
    def show_summary(self) -> None:
        """Show summary of engineered features"""
        logger.info("="*60)
        logger.info("FEATURE ENGINEERING SUMMARY")
        logger.info("="*60)
        
        logger.info(f"Final dataset: {len(self.df):,} rows × {len(self.df.columns)} columns")
        logger.info(f"Date range: {self.df['time'].min()} to {self.df['time'].max()}")
        
        # Count feature types
        feature_types = {
            'Original': len([c for c in self.df.columns if any(x in c for x in ['open_', 'high_', 'low_', 'close_', 'volume_', 'spread_'])]),
            'Log Returns': len([c for c in self.df.columns if 'log_return' in c]),
            'RSI': len([c for c in self.df.columns if 'rsi_' in c]),
            'ATR': len([c for c in self.df.columns if 'atr_' in c]),
            'MACD': len([c for c in self.df.columns if 'macd' in c]),
            'Moving Avg': len([c for c in self.df.columns if 'ma' in c and 'macd' not in c]),
            'Other': len([c for c in self.df.columns if 'price_change' in c or 'body_ratio' in c or 'dist_' in c])
        }
        
        logger.info("\nFeature breakdown:")
        for feat_type, count in feature_types.items():
            logger.info(f"  {feat_type:15s}: {count:3d} features")
        
        logger.info(f"\nTotal features: {len(self.df.columns)}")
        logger.info(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        logger.info("\nSample features (first 5):")
        print(self.df[self.df.columns[:5]].head(3).to_string(index=False))
        
        logger.info("\n" + "="*60)
        logger.info("Data ready for RL Training!")
        logger.info("="*60)
    
    def run(self) -> None:
        """Run complete feature engineering pipeline"""
        try:
            logger.info("="*60)
            logger.info("STARTING FEATURE ENGINEERING PIPELINE")
            logger.info("="*60)
            
            # 1. Load data
            self.load_data()
            logger.info("-"*60)
            
            # 2. Calculate log returns
            self.calculate_log_returns()
            logger.info("-"*60)
            
            # 3. Calculate RSI
            self.calculate_rsi(period=14)
            logger.info("-"*60)
            
            # 4. Calculate ATR (M15, H1 only)
            self.calculate_atr(period=14)
            logger.info("-"*60)
            
            # 5. Calculate MACD
            self.calculate_macd(fast=12, slow=26, signal=9)
            logger.info("-"*60)
            
            # 6. Add additional features
            self.add_additional_features()
            logger.info("-"*60)
            
            # 7. Handle NaN values
            self.handle_nan_values()
            logger.info("-"*60)
            
            # 8. Handle outliers
            self.handle_outliers(n_std=5.0)
            logger.info("-"*60)
            
            # 9. Save data
            self.save_data()
            logger.info("-"*60)
            
            # 10. Show summary
            self.show_summary()
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise


def main():
    """Main execution"""
    try:
        engineer = FeatureEngineer(
            input_file="processed_data/XAUUSD_COMBINED.parquet",
            output_file="processed_data/XAUUSD_READY_TO_TRAIN.parquet"
        )
        
        engineer.run()
        
        logger.info("\nFeature engineering completed successfully!")
        logger.info(f"Output file: {engineer.output_file}")
        
    except Exception as e:
        logger.critical(f"Critical error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

