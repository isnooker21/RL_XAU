# 🚀 Quick Start Guide - XAUUSD RL Trading Bot

เริ่มต้นฝึก AI สำหรับเทรด XAUUSD ด้วย Reinforcement Learning

---

## ✅ Prerequisites (ติดตั้งแล้ว)

- ✅ Python 3.10
- ✅ pyarrow (สำหรับอ่าน parquet)
- ✅ stable-baselines3 (PPO algorithm)
- ✅ gymnasium (RL environment)
- ✅ torch (Deep Learning)
- ✅ tensorboard (Logging)
- ✅ Data: XAUUSD_READY_TO_TRAIN.parquet (99,928 rows, 54 features)

---

## 📋 ขั้นตอนการใช้งาน (3 ขั้นตอน)

### 1️⃣ ทดสอบ Environment (เสร็จภายใน 5 วินาที)

```bash
cd /Users/isnooker/RL_XAU/xauusd_rl_project
python demo_environment.py
```

**คาดหวัง:**
- โปรแกรมจะรัน 3 episodes ด้วย random actions
- แสดงผลลัพธ์: Balance, Profit, Win Rate
- ✅ ถ้าไม่มี error แสดงว่า environment พร้อมใช้งาน

---

### 2️⃣ เริ่มการฝึก (Training)

```bash
cd /Users/isnooker/RL_XAU/xauusd_rl_project
python train_rl_model.py
```

**การตั้งค่าปัจจุบัน:**
- 🎯 Algorithm: PPO (Proximal Policy Optimization)
- 📊 Total Steps: 1,000,000 (ใช้เวลาประมาณ 2-4 ชั่วโมง)
- 💾 Save Checkpoint: ทุก 100,000 steps
- 📈 Evaluation: ทุก 50,000 steps
- 💰 Initial Balance: $10,000
- 📦 Lot Size: 0.01 (micro lot)

**ระหว่างการฝึก:**
```
Total timesteps: 1,000,000
Progress: |████████████-----| 60% (600k/1M)
Episode reward mean: 0.0234
FPS: 1500
```

**หยุดการฝึก:**
- กด `Ctrl+C` โมเดลจะบันทึกอัตโนมัติ

---

### 3️⃣ ดูผลการฝึก (TensorBoard)

**เปิดหน้าต่าง Terminal ใหม่:**

```bash
cd /Users/isnooker/RL_XAU/xauusd_rl_project
./view_tensorboard.sh
```

**เปิดเบราว์เซอร์:**
- http://localhost:6006

**Metrics ที่สำคัญ:**

| Metric | คำอธิบาย | เป้าหมาย |
|--------|----------|----------|
| `rollout/ep_rew_mean` | Reward เฉลี่ยต่อ episode | เพิ่มขึ้นเรื่อยๆ ⬆️ |
| `train/entropy_loss` | การสำรวจของ AI | ลดลงช้าๆ ⬇️ |
| `train/policy_loss` | Loss ของ policy | มีเสถียรภาพ ↔️ |
| `train/value_loss` | Loss ของ value function | ลดลงเรื่อยๆ ⬇️ |

---

## 🧪 ทดสอบโมเดลที่ฝึกเสร็จ

### หลังจากฝึกเสร็จ (หรือระหว่างฝึก)

```bash
cd /Users/isnooker/RL_XAU/xauusd_rl_project

# ทดสอบโมเดลล่าสุด
python test_model.py --model models/xauusd_model_checkpoints/xauusd_ppo_final.zip

# ทดสอบ checkpoint เฉพาะ
python test_model.py --model models/xauusd_model_checkpoints/xauusd_ppo_500000_steps.zip

# ทดสอบหลายๆ episodes
python test_model.py --model models/xauusd_model_checkpoints/xauusd_ppo_final.zip --episodes 20
```

**ผลลัพธ์:**
```
Episode 1/10:
  Steps: 1234
  Reward: 0.0456
  Final Balance: $10,456.78
  Profit: $456.78
  Trades: 45 (Win Rate: 62.2%)

...

TEST SUMMARY
==========================================
Average Reward: 0.0398 ± 0.0123
Average Profit: $398.50 ± $123.45
Average Trades: 42.3
Average Win Rate: 58.7%
Best Episode: $687.90
Worst Episode: $123.45

Plot saved: test_results_20260211_041234.png
```

---

## 📁 ไฟล์และโฟลเดอร์

```
xauusd_rl_project/
│
├── 📄 Python Scripts:
│   ├── trading_env.py           # Custom Environment (Gym)
│   ├── train_rl_model.py        # Training Script (PPO)
│   ├── test_model.py            # Test Trained Model
│   ├── demo_environment.py      # Demo/Test Environment
│   └── prepare_for_training.py  # Feature Engineering
│
├── 📂 Data:
│   └── ../processed_data/
│       └── XAUUSD_READY_TO_TRAIN.parquet  (29.8 MB)
│
├── 📂 Models (จะถูกสร้างระหว่างฝึก):
│   └── models/xauusd_model_checkpoints/
│       ├── xauusd_ppo_100000_steps.zip
│       ├── xauusd_ppo_200000_steps.zip
│       ├── ...
│       └── xauusd_ppo_final.zip
│
├── 📂 Logs (จะถูกสร้างระหว่างฝึก):
│   ├── tensorboard/            # TensorBoard logs
│   ├── monitor_train/          # Training metrics
│   ├── monitor_eval/           # Evaluation metrics
│   ├── training.log            # Training log file
│   └── feature_engineering.log # Feature engineering log
│
└── 📖 Documentation:
    ├── README_RL_TRAINING.md   # คู่มือฉบับเต็ม
    └── QUICK_START.md          # ไฟล์นี้
```

---

## 🎮 Environment Details

### Actions (4 actions)
- **0: Hold** - ไม่ทำอะไร
- **1: Buy** - เปิด Long position
- **2: Sell** - เปิด Short position
- **3: Close** - ปิด position ปัจจุบัน

### Observations (57 features)
1. **Market Features (53):**
   - OHLC ทุก timeframe (M15, H1, H4)
   - RSI, ATR, MACD
   - Moving Averages (MA20, MA50)
   - Log Returns, Price Change

2. **Position Features (4):**
   - Has position (0 or 1)
   - Position direction (-1, 0, 1)
   - Normalized profit
   - Normalized holding time

### Reward Function
```python
reward = (profit / initial_balance) + (equity_change / initial_balance)
```

- ✅ เน้น **Net Profit** จากการปิดออเดอร์
- ✅ เน้น **Equity Change** จากออเดอร์ที่เปิดอยู่
- 🔓 ไม่มี penalty (ให้ AI สำรวจอย่างอิสระ)

---

## 💡 Tips & Best Practices

### 1. ติดตามผลการฝึก
```bash
# เปิด TensorBoard พร้อมกับการฝึก (ใน Terminal ใหม่)
./view_tensorboard.sh
```

### 2. ฝึกทีละน้อยก่อน (สำหรับทดสอบ)
แก้ไขใน `train_rl_model.py`:
```python
TOTAL_TIMESTEPS = 50_000  # แทน 1,000,000
```

### 3. ปรับ Learning Rate (ถ้าเรียนรู้ช้า/เร็วเกินไป)
แก้ไขใน `train_rl_model.py`:
```python
LEARNING_RATE = 1e-4  # ลดลง (ช้าลง แต่มั่นคงขึ้น)
LEARNING_RATE = 1e-3  # เพิ่มขึ้น (เร็วขึ้น แต่อาจไม่เสถียร)
```

### 4. เพิ่ม Exploration (ถ้า AI ติดที่ action เดิมๆ)
แก้ไขใน `train_rl_model.py`:
```python
ENT_COEF = 0.05  # เพิ่มจาก 0.01
```

---

## 🔧 Troubleshooting

### ปัญหา: `ModuleNotFoundError`
```bash
pip install stable-baselines3 gymnasium torch tensorboard
```

### ปัญหา: `Data file not found`
```bash
# ตรวจสอบว่ามีไฟล์หรือไม่
ls -la ../processed_data/XAUUSD_READY_TO_TRAIN.parquet

# ถ้าไม่มี ให้รัน
cd /Users/isnooker/RL_XAU
python xauusd_rl_project/prepare_for_training.py
```

### ปัญหา: Training ช้ามาก
- ✅ ตรวจสอบ GPU: เปิด Activity Monitor → GPU Usage
- ✅ ลด `TOTAL_TIMESTEPS` เพื่อทดสอบก่อน
- ✅ ลด `N_STEPS` หรือ `BATCH_SIZE`

### ปัญหา: Reward ไม่เพิ่มขึ้น
- ✅ รอให้ฝึกไป 200k-300k steps ก่อน
- ✅ ดูกราฟใน TensorBoard (อาจขึ้นลงก่อนเสถียร)
- ✅ ปรับ `LEARNING_RATE` หรือ `ENT_COEF`

---

## 📊 Expected Training Time

| Steps | Time (approx) | Checkpoints |
|-------|---------------|-------------|
| 100k  | 15-30 min     | 1st checkpoint |
| 500k  | 1-2 hours     | Mid training |
| 1M    | 2-4 hours     | Full training |

*เวลาขึ้นอยู่กับ CPU/GPU และความเร็วของระบบ*

---

## 🎯 Next Steps (หลังจากฝึกเสร็จ)

1. **ปรับ Reward Function:**
   - เพิ่ม penalty สำหรับ drawdown มากเกินไป
   - เพิ่ม bonus สำหรับ win rate สูง

2. **ทดลอง Hyperparameters:**
   - Learning rate, batch size, entropy coefficient
   - ใช้ Optuna สำหรับ auto-tuning

3. **Backtest บนข้อมูลจริง:**
   - ทดสอบกับข้อมูลนอก training set
   - เพิ่ม slippage และ spread แบบสมจริง

4. **Deploy โมเดล:**
   - เชื่อมต่อกับ MT5 API
   - สร้าง trading bot แบบ real-time

---

## 📞 Command Summary

```bash
# 1. ทดสอบ Environment
python demo_environment.py

# 2. เริ่มการฝึก
python train_rl_model.py

# 3. ดู TensorBoard (Terminal ใหม่)
./view_tensorboard.sh

# 4. ทดสอบโมเดล
python test_model.py --model models/xauusd_model_checkpoints/xauusd_ppo_final.zip
```

---

**พร้อมแล้ว! เริ่มฝึก AI กันเลยครับ 🚀🤖📈**

```bash
cd /Users/isnooker/RL_XAU/xauusd_rl_project
python train_rl_model.py
```

