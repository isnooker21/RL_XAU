# XAUUSD Data Fetcher สำหรับ AI Training

ดึงข้อมูล XAUUSD จาก MetaTrader 5 แบบ Multi-Timeframe พร้อม Data Alignment สำหรับเทรน AI

## ⚡ Quick Start

### 1. ติดตั้ง

```bash
pip install -r requirements.txt
```

### 2. เปิด MT5 และ Login

### 3. ดึงข้อมูล

```bash
python fetch_data.py
```

### 4. เตรียมข้อมูลสำหรับเทรน AI

```bash
python prepare_for_training.py
```

เท่านี้เสร็จ! 🎉

---

## 📊 ข้อมูลที่ได้

### Timeframes
- **H4**: 10+ ปี (Trend Analysis)
- **H1**: 10+ ปี (Market Structure)
- **M15**: 5+ ปี (Entry Points)

### Output Files

```
data/                                   # Raw data แยกตาม timeframe
├── XAUUSD_H4.parquet
├── XAUUSD_H1.parquet
└── XAUUSD_M15.parquet

processed_data/                         # Aligned data + Features
├── XAUUSD_H4_aligned.parquet
├── XAUUSD_H1_aligned.parquet
├── XAUUSD_M15_aligned.parquet
├── XAUUSD_COMBINED.parquet            (Multi-TF aligned)
└── XAUUSD_READY_TO_TRAIN.parquet      ⭐ พร้อมเทรน RL (มี features ครบ)
```

---

## 🔧 Feature Engineering

`prepare_for_training.py` จะสร้าง features ต่อไปนี้:

### Features ที่เพิ่มเข้ามา:
- **Log Returns**: `log_return_M15`, `log_return_H1`, `log_return_H4`
- **RSI (14)**: `rsi_M15`, `rsi_H1`, `rsi_H4`
- **ATR (14)**: `atr_M15`, `atr_H1` (วัดความผันผวน)
- **MACD (12,26,9)**: `macd_*`, `macd_signal_*`, `macd_hist_*`
- **Moving Averages**: `ma20_*`, `ma50_*`
- **Price Change (%)**: `price_change_*`
- **Other**: `body_ratio_*`, `dist_ma20_*`

### Data Cleaning:
- ✅ ลบ NaN จากการคำนวณ indicators
- ✅ Clip outliers (5 std threshold)
- ✅ พร้อมใช้กับ RL algorithms

---

## 💻 วิธีใช้งาน

### โหลดข้อมูลเทรน RL

```python
import pandas as pd

# โหลดข้อมูลพร้อม features (แนะนำ)
df = pd.read_parquet('processed_data/XAUUSD_READY_TO_TRAIN.parquet')

print(f"Rows: {len(df):,}")
print(f"Features: {len(df.columns)}")
print(df.columns.tolist())
```

### โหลดข้อมูลพื้นฐาน

```python
# โหลดข้อมูล multi-timeframe (ยังไม่มี features)
df = pd.read_parquet('processed_data/XAUUSD_COMBINED.parquet')

# หรือโหลดแยกตาม timeframe
df_h4 = pd.read_parquet('data/XAUUSD_H4.parquet')
df_h1 = pd.read_parquet('data/XAUUSD_H1.parquet')
df_m15 = pd.read_parquet('data/XAUUSD_M15.parquet')
```

### ตัวอย่าง: Multi-Timeframe Signal

```python
import numpy as np

# โหลดข้อมูลพร้อม features
df = pd.read_parquet('processed_data/XAUUSD_READY_TO_TRAIN.parquet')

# หา Trend แต่ละ TF (ใช้ MA ที่คำนวณไว้แล้ว)
df['trend_M15'] = np.where(df['close_M15'] > df['ma20_M15'], 1, -1)
df['trend_H1'] = np.where(df['close_H1'] > df['ma20_H1'], 1, -1)
df['trend_H4'] = np.where(df['close_H4'] > df['ma20_H4'], 1, -1)

# Signal เมื่อ TF align + RSI confirm
df['strong_signal'] = (
    (df['trend_M15'] == df['trend_H1']) & 
    (df['trend_H1'] == df['trend_H4']) &
    ((df['rsi_M15'] < 30) | (df['rsi_M15'] > 70))  # RSI extreme
)

print(f"Strong signals: {df['strong_signal'].sum()}")
```

---

## ⚙️ Configuration

### เปลี่ยน Symbol

แก้ไขใน `fetch_data.py` บรรทัดที่ 41:

```python
SYMBOL = "XAUUSD"  # เปลี่ยนเป็น "XAUUSD.s", "GOLDm" ตามที่ Broker ใช้
```

สคริปต์จะ auto-detect suffix อัตโนมัติ

### เปลี่ยนจำนวนปี

แก้ไขใน `fetch_data.py` บรรทัดที่ 44-48:

```python
TIMEFRAMES = {
    'H4': {'mt5': mt5.TIMEFRAME_H4, 'years': 15},  # เปลี่ยนเป็น 15 ปี
    'H1': {'mt5': mt5.TIMEFRAME_H1, 'years': 12},  # เปลี่ยนเป็น 12 ปี
    'M15': {'mt5': mt5.TIMEFRAME_M15, 'years': 7}  # เปลี่ยนเป็น 7 ปี
}
```

---

## 🎯 ทำไมต้อง Multi-Timeframe?

### ปัญหาของการใช้ Timeframe เดียว

```python
# AI มองเห็นแค่ M15 ⚠️
df = load('M15.parquet')
# ไม่รู้ว่า H4 trend เป็นยังไง!
```

### แก้ด้วย Multi-Timeframe

```python
# AI เห็นทุก timeframe พร้อมกัน ✅
df = load('XAUUSD_COMBINED.parquet')

# ตัวอย่าง: M15 bar ที่ 10:15
# - M15: ข้อมูล 10:15 (entry level)
# - H1:  ข้อมูล 10:00 (market structure)
# - H4:  ข้อมูล 08:00 (main trend)
```

**Result**: AI เห็นภาพรวมเหมือนนักเทรดมืออาชีพ!

---

## 🔧 Troubleshooting

### Symbol not found

```python
# ลองเปลี่ยน symbol ใน fetch_data.py
SYMBOL = "XAUUSD.s"  # หรือ "XAUUSD.i", "GOLDm"
```

### Failed to connect MT5

1. เปิด MT5 Terminal
2. Login เข้า Account
3. ลองปิด-เปิด MT5 ใหม่

### Not enough data

บาง Broker มีข้อมูลไม่ครบ 10 ปี ลด years target ลง:

```python
'H4': {'mt5': mt5.TIMEFRAME_H4, 'years': 5}  # ลดเหลือ 5 ปี
```

---

## 📋 Requirements

- Python 3.8+
- MetaTrader 5 Terminal (Windows only)
- pandas, pyarrow

---

## 📦 Repository

**GitHub**: https://github.com/isnooker21/RL_XAU

---

**พร้อมเทรน AI!** 🚀
