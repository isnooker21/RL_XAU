# Quick Start Guide - Multi-Timeframe Data Fetcher

## 🚀 เริ่มต้นใน 3 นาที

### Step 1: ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

**หมายเหตุ:** MetaTrader5 package ใช้ได้เฉพาะ Windows เท่านั้น

### Step 2: เปิด MT5 และ Login

1. เปิด MetaTrader 5 Terminal
2. Login เข้า Account ของคุณ
3. ตรวจสอบว่า XAUUSD symbol มีอยู่ใน Market Watch

### Step 3: ดึงข้อมูล Multi-Timeframe

```bash
cd xauusd_rl_project
python fetch_multi_timeframe.py
```

### Step 4: ตรวจสอบผลลัพธ์

```
✅ data/
   ├── XAUUSD_H4.parquet    (Raw H4 data)
   ├── XAUUSD_H1.parquet    (Raw H1 data)
   └── XAUUSD_M15.parquet   (Raw M15 data)

✅ processed_data/
   ├── XAUUSD_H4_aligned.parquet
   ├── XAUUSD_H1_aligned.parquet
   ├── XAUUSD_M15_aligned.parquet
   └── XAUUSD_COMBINED.parquet  ⭐ Use this for AI!
```

---

## 📊 วิธีใช้งานข้อมูล (2 บรรทัด!)

```python
import pandas as pd

# โหลดข้อมูล multi-timeframe
df = pd.read_parquet('processed_data/XAUUSD_COMBINED.parquet')

# พร้อมใช้งาน!
print(f"Total bars: {len(df):,}")
print(df.tail())
```

### ตัวอย่าง Output:

```
Total bars: 95,234

time                  close_M15  close_H1  close_H4
2026-02-10 19:00:00    2850.50   2850.50   2848.25
2026-02-10 19:15:00    2851.20   2850.50   2848.25
2026-02-10 19:30:00    2849.80   2850.50   2848.25
2026-02-10 19:45:00    2850.90   2850.90   2848.25
2026-02-10 20:00:00    2851.50   2851.50   2848.25
```

---

## 🧠 ทำไมต้องใช้ Multi-Timeframe?

### ❌ แบบเก่า (Single Timeframe)

```python
# AI มองเห็นแค่ M15 bar ปัจจุบัน
df_m15 = pd.read_parquet('data/XAUUSD_M15.parquet')
# AI ไม่รู้ว่า H4 กำลังขาขึ้นหรือขาลง!
```

### ✅ แบบใหม่ (Multi-Timeframe Aligned)

```python
# AI มองเห็นทั้ง H4 trend, H1 structure, และ M15 entry พร้อมกัน!
df = pd.read_parquet('processed_data/XAUUSD_COMBINED.parquet')

# ตัวอย่าง: M15 bar ที่ 19:15 จะมีข้อมูล
# - M15: ข้อมูล 19:15 (entry level)
# - H1:  ข้อมูล 19:00 (market structure)
# - H4:  ข้อมูล 16:00 (main trend)
```

**ผลลัพธ์:** AI เห็นภาพรวมเหมือนนักเทรดมืออาชีพ!

---

## 🎯 Use Cases

### 1. Trend Following AI

```python
# AI เรียนรู้ว่า: ถ้า H4 ขาขึ้น + H1 breakout + M15 pullback = BUY!
features = ['close_H4', 'close_H1', 'close_M15', 'rsi_M15']
```

### 2. Reversal Detection AI

```python
# AI เรียนรู้ว่า: ถ้า H4 overbought + H1 reversal pattern = SELL!
features = ['rsi_H4', 'rsi_H1', 'close_M15', 'volume_M15']
```

### 3. Smart Position Sizing

```python
# ใช้ ATR จาก H4 เพื่อกำหนดขนาด position
df['position_size'] = 1000 / df['atr_H4']
```

---

## 📚 Next Steps

1. **อ่านคู่มือเต็ม:** [MULTI_TIMEFRAME_GUIDE.md](MULTI_TIMEFRAME_GUIDE.md)
2. **ดู Examples:** รัน `python example_load_data.py`
3. **เทรน AI Model:** ใช้ `processed_data/XAUUSD_COMBINED.parquet`

---

## ❓ Troubleshooting

### "Symbol XAUUSD not found"

แก้ไขใน `fetch_multi_timeframe.py`:

```python
SYMBOL = "XAUUSD.s"  # หรือ "XAUUSD.i", "GOLDm" ตามที่ Broker ใช้
```

### "Failed to connect to MT5"

1. ตรวจสอบว่า MT5 Terminal เปิดอยู่
2. ตรวจสอบว่า Login สำเร็จแล้ว
3. ลองปิด-เปิด MT5 ใหม่

### "Not enough historical data"

บาง Broker มีข้อมูลย้อนหลังไม่ครบ 10 ปี ลองลด years target:

```python
TIMEFRAMES = {
    'H4': {'mt5': mt5.TIMEFRAME_H4, 'years': 5},  # ลดเหลือ 5 ปี
    ...
}
```

---

## 🎉 Done!

คุณพร้อมเทรน AI แบบ Multi-Timeframe Analysis แล้ว!

**Repository:** https://github.com/isnooker21/RL_XAU

**Questions?** เปิด Issue บน GitHub หรือติดต่อผู้พัฒนา

