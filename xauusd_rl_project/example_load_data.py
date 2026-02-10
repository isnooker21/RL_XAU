"""
Example: How to Load and Use Multi-Timeframe Data
==================================================
This script demonstrates how to load and analyze the aligned multi-timeframe data
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_combined_dataset(symbol: str = "XAUUSD") -> pd.DataFrame:
    """Load the combined multi-timeframe dataset"""
    filepath = Path("processed_data") / f"{symbol}_COMBINED.parquet"
    
    if not filepath.exists():
        # Try with suffix
        files = list(Path("processed_data").glob(f"{symbol}*_COMBINED.parquet"))
        if files:
            filepath = files[0]
        else:
            raise FileNotFoundError(f"Combined dataset not found for {symbol}")
    
    print(f"Loading: {filepath}")
    df = pd.read_parquet(filepath)
    
    print(f"\nDataset Info:")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Date Range: {df['time'].min()} to {df['time'].max()}")
    print(f"  Time Span: {(df['time'].max() - df['time'].min()).days} days")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    return df


def load_individual_timeframe(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Load individual aligned timeframe
    
    Args:
        symbol: Trading symbol (e.g., 'XAUUSD')
        timeframe: 'M15', 'H1', or 'H4'
    """
    filepath = Path("processed_data") / f"{symbol}_{timeframe}_aligned.parquet"
    
    if not filepath.exists():
        # Try with suffix
        files = list(Path("processed_data").glob(f"{symbol}*_{timeframe}_aligned.parquet"))
        if files:
            filepath = files[0]
        else:
            raise FileNotFoundError(f"{timeframe} aligned dataset not found")
    
    print(f"Loading: {filepath}")
    df = pd.read_parquet(filepath)
    print(f"  {timeframe}: {len(df):,} bars")
    
    return df


def example_multi_timeframe_analysis():
    """Example: Analyze multi-timeframe data"""
    print("="*60)
    print("EXAMPLE: Multi-Timeframe Analysis")
    print("="*60)
    
    # Load combined dataset
    df = load_combined_dataset()
    
    print("\n" + "="*60)
    print("AVAILABLE COLUMNS")
    print("="*60)
    print("\nM15 Columns (Entry Level):")
    m15_cols = [col for col in df.columns if '_M15' in col]
    print(f"  {', '.join(m15_cols)}")
    
    print("\nH1 Columns (Market Structure):")
    h1_cols = [col for col in df.columns if '_H1' in col]
    print(f"  {', '.join(h1_cols)}")
    
    print("\nH4 Columns (Trend Analysis):")
    h4_cols = [col for col in df.columns if '_H4' in col]
    print(f"  {', '.join(h4_cols)}")
    
    # Example: Compare trends across timeframes
    print("\n" + "="*60)
    print("EXAMPLE: Recent Price Comparison")
    print("="*60)
    
    recent = df.tail(5)[['time', 'close_M15', 'close_H1', 'close_H4']]
    print(recent.to_string(index=False))
    
    # Example: Calculate multi-timeframe moving averages
    print("\n" + "="*60)
    print("EXAMPLE: Multi-Timeframe Moving Averages")
    print("="*60)
    
    df['ma20_M15'] = df['close_M15'].rolling(20).mean()
    df['ma20_H1'] = df['close_H1'].rolling(20).mean()
    df['ma20_H4'] = df['close_H4'].rolling(20).mean()
    
    print("\nLatest Moving Averages:")
    latest = df.tail(1)[['time', 'ma20_M15', 'ma20_H1', 'ma20_H4']]
    print(latest.to_string(index=False))
    
    # Example: Multi-timeframe trend detection
    print("\n" + "="*60)
    print("EXAMPLE: Trend Alignment Detection")
    print("="*60)
    
    df['trend_M15'] = np.where(df['close_M15'] > df['ma20_M15'], 'UP', 'DOWN')
    df['trend_H1'] = np.where(df['close_H1'] > df['ma20_H1'], 'UP', 'DOWN')
    df['trend_H4'] = np.where(df['close_H4'] > df['ma20_H4'], 'UP', 'DOWN')
    
    # Find bars where all timeframes align
    df['all_aligned'] = (
        (df['trend_M15'] == df['trend_H1']) & 
        (df['trend_H1'] == df['trend_H4'])
    )
    
    aligned_count = df['all_aligned'].sum()
    aligned_pct = (aligned_count / len(df)) * 100
    
    print(f"Bars with aligned trends: {aligned_count:,} ({aligned_pct:.2f}%)")
    
    recent_aligned = df.tail(10)[['time', 'trend_M15', 'trend_H1', 'trend_H4', 'all_aligned']]
    print("\nRecent Trend Alignment:")
    print(recent_aligned.to_string(index=False))
    
    return df


def example_prepare_for_ml():
    """Example: Prepare data for Machine Learning"""
    print("\n" + "="*60)
    print("EXAMPLE: Prepare for ML Training")
    print("="*60)
    
    df = load_combined_dataset()
    
    # Select features for ML
    feature_cols = [
        # M15 features (current bar)
        'open_M15', 'high_M15', 'low_M15', 'close_M15', 'tick_volume_M15',
        # H1 context
        'open_H1', 'high_H1', 'low_H1', 'close_H1', 'tick_volume_H1',
        # H4 trend
        'open_H4', 'high_H4', 'low_H4', 'close_H4', 'tick_volume_H4'
    ]
    
    # Create feature matrix
    X = df[feature_cols].copy()
    
    # Add technical indicators across timeframes
    for tf in ['M15', 'H1', 'H4']:
        close_col = f'close_{tf}'
        high_col = f'high_{tf}'
        low_col = f'low_{tf}'
        
        # Moving averages
        X[f'ma20_{tf}'] = df[close_col].rolling(20).mean()
        X[f'ma50_{tf}'] = df[close_col].rolling(50).mean()
        
        # RSI
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        X[f'rsi_{tf}'] = 100 - (100 / (1 + rs))
        
        # ATR
        hl = df[high_col] - df[low_col]
        hc = abs(df[high_col] - df[close_col].shift())
        lc = abs(df[low_col] - df[close_col].shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        X[f'atr_{tf}'] = tr.rolling(14).mean()
    
    # Remove rows with NaN (from indicators)
    X = X.dropna()
    
    print(f"\nFeature Matrix Shape: {X.shape}")
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]:,}")
    
    print("\nFeature Categories:")
    print(f"  M15 features: {len([c for c in X.columns if 'M15' in c])}")
    print(f"  H1 features:  {len([c for c in X.columns if 'H1' in c])}")
    print(f"  H4 features:  {len([c for c in X.columns if 'H4' in c])}")
    
    print("\nSample features (first 5 columns):")
    print(X.head()[X.columns[:5]])
    
    print("\n" + "="*60)
    print("Data ready for ML training!")
    print("="*60)
    
    return X


def main():
    """Run all examples"""
    try:
        # Example 1: Multi-timeframe analysis
        df = example_multi_timeframe_analysis()
        
        # Example 2: Prepare for ML
        X = example_prepare_for_ml()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run 'fetch_multi_timeframe.py' first to generate the data.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

