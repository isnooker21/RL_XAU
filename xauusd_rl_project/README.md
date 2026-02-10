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

เท่านี้เสร็จ! 🎉

---

## 📊 ข้อมูลที่ได้

### Timeframes
- **H4**: 10+ ปี (Trend Analysis)
- **H1**: 10+ ปี (Market Structure)
- **M15**: 5+ ปี (Entry Points)

### Output Files

```
data/                          # Raw data แยกตาม timeframe
├── XAUUSD_H4.parquet
├── XAUUSD_H1.parquet
└── XAUUSD_M15.parquet

processed_data/                # Aligned data พร้อมใช้
├── XAUUSD_H4_aligned.parquet
├── XAUUSD_H1_aligned.parquet
├── XAUUSD_M15_aligned.parquet
└── XAUUSD_COMBINED.parquet    ⭐ ใช้ไฟล์นี้เทรน AI
```

---

## 💻 วิธีใช้งาน

### โหลดข้อมูล

```python
import pandas as pd

# โหลดข้อมูล multi-timeframe (แนะนำ)
df = pd.read_parquet('processed_data/XAUUSD_COMBINED.parquet')

# หรือโหลดแยกตาม timeframe
df_h4 = pd.read_parquet('data/XAUUSD_H4.parquet')
df_h1 = pd.read_parquet('data/XAUUSD_H1.parquet')
df_m15 = pd.read_parquet('data/XAUUSD_M15.parquet')
```

### โครงสร้างข้อมูล COMBINED

```python
# แต่ละ row มีข้อมูลทุก timeframe พร้อมกัน
print(df.columns)
# ['time', 'open_M15', 'high_M15', 'low_M15', 'close_M15', ...,
#  'open_H1', 'high_H1', 'low_H1', 'close_H1', ...,
#  'open_H4', 'high_H4', 'low_H4', 'close_H4', ...]

# AI มองเห็นทุก timeframe พร้อมกัน!
print(df.tail())
```

### ตัวอย่าง: Multi-Timeframe Analysis

```python
import numpy as np

# คำนวณ Moving Average ทุก TF
df['ma20_M15'] = df['close_M15'].rolling(20).mean()
df['ma20_H1'] = df['close_H1'].rolling(20).mean()
df['ma20_H4'] = df['close_H4'].rolling(20).mean()

# หา Trend แต่ละ TF
df['trend_M15'] = np.where(df['close_M15'] > df['ma20_M15'], 'UP', 'DOWN')
df['trend_H1'] = np.where(df['close_H1'] > df['ma20_H1'], 'UP', 'DOWN')
df['trend_H4'] = np.where(df['close_H4'] > df['ma20_H4'], 'UP', 'DOWN')

# Signal เมื่อทุก TF align กัน
df['all_aligned'] = (
    (df['trend_M15'] == df['trend_H1']) & 
    (df['trend_H1'] == df['trend_H4'])
)
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
