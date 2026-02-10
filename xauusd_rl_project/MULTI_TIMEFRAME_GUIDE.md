# Multi-Timeframe Data Fetcher Guide

## 🎯 Overview

สคริปต์นี้ออกแบบมาเพื่อดึงข้อมูล **3 Timeframes พร้อมกัน** และทำ **Data Alignment** เพื่อให้ AI สามารถมองเห็นภาพรวมของตลาดในหลายมิติพร้อมกัน

### Strategy

```
H4  → Trend Analysis     (เป้าหมาย 10+ ปี)
H1  → Market Structure   (เป้าหมาย 10+ ปี)
M15 → Entry Points       (เป้าหมาย 5+ ปี)
```

## 🚀 Quick Start

### 1. ดึงข้อมูลแบบ Multi-Timeframe

```bash
# รันสคริปต์หลัก
python fetch_multi_timeframe.py
```

### 2. ผลลัพธ์ที่ได้

```
data/                           # Raw data (แยกตาม timeframe)
├── XAUUSD_H4.parquet
├── XAUUSD_H1.parquet
└── XAUUSD_M15.parquet

processed_data/                 # Aligned data (พร้อมใช้กับ AI)
├── XAUUSD_H4_aligned.parquet
├── XAUUSD_H1_aligned.parquet
├── XAUUSD_M15_aligned.parquet
└── XAUUSD_COMBINED.parquet     ⭐ ไฟล์หลักสำหรับ AI Training
```

## 📊 Data Alignment Strategy

### ปัญหาที่แก้ไข

ปกติแต่ละ timeframe มี timestamps ที่ไม่ตรงกัน:

```
H4:   00:00, 04:00, 08:00, 12:00, 16:00, 20:00
H1:   00:00, 01:00, 02:00, 03:00, 04:00, ...
M15:  00:00, 00:15, 00:30, 00:45, 01:00, ...
```

### วิธีแก้

ใช้ **M15 เป็น Base Timeline** และ **Forward-fill** ข้อมูล H1 และ H4:

```python
# ตัวอย่าง: M15 bar ที่ 10:15 จะได้ข้อมูล
- M15: ข้อมูล bar 10:15
- H1:  ข้อมูล bar 10:00 (ล่าสุดก่อนหน้า)
- H4:  ข้อมูล bar 08:00 (ล่าสุดก่อนหน้า)
```

ทำให้ AI มองเห็น **Context ของ Higher Timeframe** ในทุก M15 bar!

## 📁 Combined Dataset Structure

ไฟล์ `XAUUSD_COMBINED.parquet` มีโครงสร้างดังนี้:

| time | open_M15 | high_M15 | low_M15 | close_M15 | ... | open_H1 | high_H1 | ... | open_H4 | high_H4 | ... |
|------|----------|----------|---------|-----------|-----|---------|---------|-----|---------|---------|-----|
| 2024-01-01 00:00 | 2050.5 | 2051.2 | 2049.8 | 2050.9 | ... | 2048.5 | 2052.0 | ... | 2045.0 | 2055.0 | ... |
| 2024-01-01 00:15 | 2050.9 | 2051.5 | 2050.1 | 2051.2 | ... | 2048.5 | 2052.0 | ... | 2045.0 | 2055.0 | ... |

**คุณสมบัติ:**
- แต่ละ row = M15 bar หนึ่งอัน
- แต่ละ row มีข้อมูล H1 และ H4 ที่เกี่ยวข้อง
- AI เห็นภาพรวมทุก timeframe พร้อมกัน

## 💻 วิธีใช้งานข้อมูล

### โหลดข้อมูล Combined

```python
import pandas as pd

# โหลดข้อมูล multi-timeframe
df = pd.read_parquet('processed_data/XAUUSD_COMBINED.parquet')

print(f"Total bars: {len(df):,}")
print(f"Columns: {len(df.columns)}")

# ดูข้อมูลล่าสุด
print(df.tail())
```

### แยกข้อมูลตาม Timeframe

```python
# M15 columns
m15_cols = [col for col in df.columns if '_M15' in col]
df_m15 = df[['time'] + m15_cols]

# H1 columns
h1_cols = [col for col in df.columns if '_H1' in col]
df_h1 = df[['time'] + h1_cols]

# H4 columns
h4_cols = [col for col in df.columns if '_H4' in col]
df_h4 = df[['time'] + h4_cols]
```

### ตัวอย่าง: Multi-Timeframe Trend Detection

```python
import numpy as np

# คำนวณ Moving Average ทุก timeframe
df['ma20_M15'] = df['close_M15'].rolling(20).mean()
df['ma20_H1'] = df['close_H1'].rolling(20).mean()
df['ma20_H4'] = df['close_H4'].rolling(20).mean()

# หา Trend แต่ละ timeframe
df['trend_M15'] = np.where(df['close_M15'] > df['ma20_M15'], 1, -1)
df['trend_H1'] = np.where(df['close_H1'] > df['ma20_H1'], 1, -1)
df['trend_H4'] = np.where(df['close_H4'] > df['ma20_H4'], 1, -1)

# Signal เมื่อ Trend ทุก TF ชี้ทิศทางเดียวกัน
df['all_aligned'] = (
    (df['trend_M15'] == df['trend_H1']) & 
    (df['trend_H1'] == df['trend_H4'])
)

# Filter เฉพาะ bars ที่ align กัน
aligned_bars = df[df['all_aligned'] == True]
print(f"Aligned bars: {len(aligned_bars):,}")
```

### ตัวอย่าง: เตรียมข้อมูลสำหรับ ML

```python
from sklearn.preprocessing import StandardScaler

# Select features
features = [
    # M15 OHLC
    'open_M15', 'high_M15', 'low_M15', 'close_M15',
    # H1 OHLC
    'open_H1', 'high_H1', 'low_H1', 'close_H1',
    # H4 OHLC
    'open_H4', 'high_H4', 'low_H4', 'close_H4',
]

X = df[features].copy()

# Add technical indicators
for tf in ['M15', 'H1', 'H4']:
    X[f'ma20_{tf}'] = df[f'close_{tf}'].rolling(20).mean()
    X[f'rsi_{tf}'] = calculate_rsi(df[f'close_{tf}'])

# Remove NaN and normalize
X = X.dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]:,}")
```

## 🔧 Configuration

### เปลี่ยน Symbol

แก้ไขใน `fetch_multi_timeframe.py`:

```python
class MultiTimeframeFetcher:
    SYMBOL = "XAUUSD"  # เปลี่ยนเป็น "GOLDm", "XAUUSD.i", etc.
```

สคริปต์จะ auto-detect suffix (.s, .i, etc.) อัตโนมัติ

### เปลี่ยนเป้าหมายจำนวนปี

แก้ไขใน `fetch_multi_timeframe.py`:

```python
TIMEFRAMES = {
    'H4': {'mt5': mt5.TIMEFRAME_H4, 'years': 15},  # เปลี่ยนเป็น 15 ปี
    'H1': {'mt5': mt5.TIMEFRAME_H1, 'years': 12},  # เปลี่ยนเป็น 12 ปี
    'M15': {'mt5': mt5.TIMEFRAME_M15, 'years': 7}, # เปลี่ยนเป็น 7 ปี
}
```

### เพิ่ม Timeframes อื่น

```python
TIMEFRAMES = {
    'D1': {'mt5': mt5.TIMEFRAME_D1, 'years': 20, 'desc': 'Long-term Trend'},
    'H4': {'mt5': mt5.TIMEFRAME_H4, 'years': 10, 'desc': 'Trend Analysis'},
    'H1': {'mt5': mt5.TIMEFRAME_H1, 'years': 10, 'desc': 'Market Structure'},
    'M15': {'mt5': mt5.TIMEFRAME_M15, 'years': 5, 'desc': 'Entry Points'},
    'M5': {'mt5': mt5.TIMEFRAME_M5, 'years': 2, 'desc': 'Scalping'},
}
```

## 📈 Use Cases for AI Training

### 1. **Trend Following Strategy**

```python
# Features: H4 trend + H1 structure + M15 entry
features = [
    'close_H4', 'ma20_H4', 'ma50_H4',  # Trend
    'close_H1', 'ma20_H1',              # Structure
    'close_M15', 'rsi_M15'              # Entry timing
]
```

### 2. **Reversal Detection**

```python
# Features: ดู divergence ระหว่าง timeframes
df['h4_h1_divergence'] = df['close_H4'] - df['close_H1']
df['h1_m15_divergence'] = df['close_H1'] - df['close_M15']
```

### 3. **Volatility-based Position Sizing**

```python
# ใช้ ATR จากหลาย timeframes
df['atr_h4'] = calculate_atr(df, 'H4', 14)
df['atr_h1'] = calculate_atr(df, 'H1', 14)
df['atr_m15'] = calculate_atr(df, 'M15', 14)

# Position size = f(ATR_H4, ATR_H1)
```

## ⚠️ Important Notes

### Missing Data Handling

- Script ใช้ **forward-fill** สำหรับ missing bars
- H4/H1 bars จะถูก carry forward ไปยัง M15 bars ถัดไป
- เหมาะสำหรับ AI training (AI เห็น context ล่าสุด)

### Memory Considerations

Combined dataset อาจใช้ memory มาก:

```
M15 (5 years) × (7 columns × 3 timeframes) = ~21 columns
Approx 100,000 rows × 21 columns × 8 bytes = ~17 MB
```

สำหรับ 10 ปีของ M15 จะใช้ประมาณ 30-40 MB

### Broker Limitations

ข้อมูลที่ได้จริงขึ้นกับ:
1. Broker history availability
2. MT5 API limits (~100k bars per fetch)
3. Symbol availability

## 🎓 Advanced: Running Example

```bash
# ดึงข้อมูล
python fetch_multi_timeframe.py

# รัน examples
python example_load_data.py
```

`example_load_data.py` จะแสดง:
- วิธีโหลดข้อมูล
- Multi-timeframe trend analysis
- Feature engineering examples
- ML data preparation

## 📚 References

- **Timeframe Correlation**: H4 trend → H1 confirmation → M15 entry
- **Forward Fill Logic**: Use most recent higher TF bar
- **Alignment Strategy**: M15 as base timeline

---

**Ready for AI Training!** 🚀🤖

สคริปต์นี้ช่วยให้ AI มองเห็น "Big Picture" และ "Detail" พร้อมกัน เหมือนนักเทรดมืออาชีพที่ดูหลาย timeframes ก่อนตัดสินใจ!

