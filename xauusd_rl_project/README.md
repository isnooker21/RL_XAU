# XAUUSD Historical Data Fetcher

Python Scripts สำหรับดึงข้อมูล Historical Data ของ XAUUSD จาก MetaTrader 5 เพื่อใช้ในการเทรน AI/ML Models

## 🎯 Available Scripts

### 1. **Multi-Timeframe Fetcher** ⭐ (แนะนำสำหรับ AI Training)

```bash
python fetch_multi_timeframe.py
```

**Features:**
- 📊 ดึง 3 Timeframes พร้อมกัน: **H4 (Trend)** + **H1 (Structure)** + **M15 (Entry)**
- 🎯 เป้าหมาย: H4/H1 = 10+ ปี, M15 = 5+ ปี
- 🔄 **Data Alignment**: ทุก timeframe มี timestamp ที่ตรงกัน
- 🧠 **Combined Dataset**: AI มองเห็นทุก timeframe พร้อมกัน
- 📁 Export ทั้ง Raw และ Processed data

👉 [**อ่านคู่มือฉบับเต็ม: MULTI_TIMEFRAME_GUIDE.md**](MULTI_TIMEFRAME_GUIDE.md)

### 2. Basic Fetcher (4 Timeframes)

```bash
python fetch_xauusd_data.py
```

**Features:**
- 📊 ดึงข้อมูล 4 Timeframes: M1, M5, M15, H1
- 📅 ดึงข้อมูลย้อนหลัง 3-5 ปี
- 💾 บันทึกเป็น Parquet format แยกไฟล์

### 3. Multi-Batch Fetcher (สำหรับ M1/M5)

```bash
python fetch_multi_batch.py
```

**Features:**
- 🔄 ดึงข้อมูล M1/M5 มากกว่า 100k bars
- 📦 แบ่งดึงเป็นช่วงๆ แล้วรวมกัน

## ✨ Core Features (ทุก Scripts)

- 🔌 เชื่อมต่อ MT5 Terminal อย่างปลอดภัย พร้อม Auto-disconnect
- 🧹 Data Cleaning: แปลง timestamp, เลือก columns ที่สำคัญ
- 💾 บันทึกเป็น Parquet format พร้อม Snappy compression
- 📝 Logging ครบถ้วน: จำนวนแท่ง, ช่วงเวลา, ขนาดไฟล์
- ⚠️ Error Handling แบบครอบคลุม พร้อม Fallback mechanism
- 🪟 Windows-compatible: รองรับ Unicode encoding

## 📋 Requirements

- Python 3.8+
- MetaTrader 5 Terminal (เปิดและ Login อยู่)
- Account: 2000730 (Kasidit Sangsipet)

## 🚀 Installation

```bash
# 1. ติดตั้ง dependencies
pip install -r requirements.txt

# 2. เปิด MT5 Terminal และ Login
# 3. รันสคริปต์
python fetch_xauusd_data.py

# (Optional) หากต้องการข้อมูล M1/M5 ย้อนหลังมากกว่า 100,000 bars
python fetch_multi_batch.py
```

## ⚙️ Configuration

หากต้องการปรับจำนวนปีที่ดึงข้อมูล แก้ไขใน `main()` function:

```python
# เปลี่ยน years_back=5 เป็นจำนวนปีที่ต้องการ (เช่น 3, 7, 10)
with MT5DataFetcher(data_dir="data", years_back=5) as fetcher:
    ...
```

## 📁 Output Files

ไฟล์จะถูกบันทึกในโฟลเดอร์ `data/`:

```
data/
├── XAUUSD_M1.parquet   # ข้อมูล 1 นาที
├── XAUUSD_M5.parquet   # ข้อมูล 5 นาที
├── XAUUSD_M15.parquet  # ข้อมูล 15 นาที
└── XAUUSD_H1.parquet   # ข้อมูル 1 ชั่วโมง
```

Log file: `xauusd_data_fetch.log`

## 📊 Data Schema

แต่ละไฟล์ Parquet จะมี columns ดังนี้:

| Column       | Type      | Description                |
|-------------|-----------|----------------------------|
| time        | datetime  | Timestamp ของแท่งเทียน     |
| open        | float64   | ราคาเปิด                   |
| high        | float64   | ราคาสูงสุด                 |
| low         | float64   | ราคาต่ำสุด                 |
| close       | float64   | ราคาปิด                    |
| tick_volume | int64     | จำนวน ticks                |
| spread      | int64     | Spread (in points)         |
| real_volume | int64     | Real trading volume        |

## 🔍 Example Usage

### โหลดข้อมูลเพื่อใช้งาน:

```python
import pandas as pd

# อ่านข้อมูล M1
df_m1 = pd.read_parquet('data/XAUUSD_M1.parquet')

print(f"Total bars: {len(df_m1):,}")
print(f"Date range: {df_m1['time'].min()} to {df_m1['time'].max()}")
print(df_m1.head())
```

### ตรวจสอบขนาดข้อมูล:

```python
import os

for tf in ['M1', 'M5', 'M15', 'H1']:
    filepath = f'data/XAUUSD_{tf}.parquet'
    if os.path.exists(filepath):
        df = pd.read_parquet(filepath)
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        days = (df['time'].max() - df['time'].min()).days
        print(f"{tf}: {len(df):,} bars, {days} days, {size_mb:.2f} MB")
```

## 📝 Log Output Example

```
2026-02-11 10:30:15 - INFO - Connected to MT5 Terminal
2026-02-11 10:30:15 - INFO - Account: 2000730: Kasidit Sangsipet
2026-02-11 10:30:15 - INFO - ============================================================
2026-02-11 10:30:15 - INFO - Starting XAUUSD Historical Data Download
2026-02-11 10:30:15 - INFO - Target: 5 years of historical data
2026-02-11 10:30:15 - INFO - ============================================================
2026-02-11 10:30:15 - INFO - Fetching M1 data for XAUUSD...
2026-02-11 10:30:15 - INFO -   Requested range: 2021-02-11 to 2026-02-11 (5 years)
2026-02-11 10:30:16 - INFO - ✓ M1: Fetched 1,234,567 bars
2026-02-11 10:30:16 - INFO -   Actual range: 2021-02-11 00:00:00 to 2026-02-11 10:30:00
2026-02-11 10:30:16 - INFO -   Time span: 1,826 days (5.00 years)
2026-02-11 10:30:16 - INFO -   Memory usage: 85.42 MB
2026-02-11 10:30:17 - INFO - ✓ Saved to data/XAUUSD_M1.parquet (28.15 MB)
```

## ⚠️ Important Notes

### MT5 API Limitations

**MT5 จำกัดการดึงข้อมูลครั้งละไม่เกิน ~100,000 bars** ดังนั้น:

| Timeframe | 5 ปี = bars | ที่ดึงได้จริง | ระยะเวลาจริง |
|-----------|-------------|--------------|--------------|
| M1 | ~1.8 ล้าน | ~100,000 | ~3-4 เดือน |
| M5 | ~350,000 | ~100,000 | ~1-1.5 ปี |
| M15 | ~120,000 | ~100,000 | ~4 ปี |
| H1 | ~30,000 | ~30,000 | **5 ปีเต็ม** ✅ |

**วิธีแก้:** ใช้ `fetch_multi_batch.py` เพื่อดึงข้อมูล M1/M5 แบบแบ่งเป็นช่วงๆ

### General Notes

1. **MT5 Terminal ต้องเปิดอยู่** และ Login ก่อนรันสคริปต์
2. **ข้อมูลขึ้นกับ Broker** - บาง Broker มีข้อมูลย้อนหลังไม่ครบ
3. **Symbol Name** - บาง Broker ใช้ `XAUUSD.s`, `XAUUSD.i`, หรือ `GOLDm` (แก้ไขใน script)
4. **H1 และ M15 เหมาะสำหรับ AI Training มากที่สุด** - ข้อมูลครบและ noise น้อย

## 🔄 Multi-Batch Fetcher (Advanced)

หากต้องการดึงข้อมูล **M1/M5 ย้อนหลังเต็ม 5 ปี** (มากกว่า 100,000 bars):

```bash
python fetch_multi_batch.py
```

### วิธีการทำงาน:
1. แบ่งดึงข้อมูลเป็นช่วงๆ (M1: ทีละ 90 วัน, M5: ทีละ 180 วัน)
2. รวมข้อมูลทั้งหมดและลบข้อมูลซ้ำ
3. บันทึกเป็นไฟล์ `XAUUSD.s_M1_FULL.parquet` และ `XAUUSD.s_M5_FULL.parquet`

### ตัวอย่างผลลัพธ์:
- **M1**: อาจได้มากกว่า 500,000 bars (หลายเดือนถึง 1-2 ปี)
- **M5**: อาจได้มากกว่า 200,000 bars (2-3 ปี)

**หมายเหตุ:** ระยะเวลาที่ได้จริงขึ้นกับข้อมูลที่ Broker เก็บไว้

## 🛠️ Troubleshooting

### "Failed to connect to MT5"
- ตรวจสอบว่า MT5 Terminal เปิดอยู่และ Login แล้ว
- ลองปิด-เปิด MT5 ใหม่

### "No data available"
- Broker อาจไม่มีข้อมูลย้อนหลังครบ 5 ปี
- ลองลด `years_back` เป็น 3 หรือ 2 ปี

### "Symbol XAUUSD not found"
- ตรวจสอบชื่อ Symbol ใน MT5 (บาง Broker ใช้ชื่อต่างกัน เช่น XAUUSD.i, GOLDm)
- แก้ไข `SYMBOL = "XAUUSD"` ใน class `MT5DataFetcher`

## 📧 Support

Account: 2000730 - Kasidit Sangsipet

---

**Ready for AI Training! 🚀**

