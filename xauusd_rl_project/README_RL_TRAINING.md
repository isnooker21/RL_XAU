# XAUUSD RL Trading Bot - Training Guide
การฝึก AI สำหรับเทรด XAUUSD ด้วย Reinforcement Learning (PPO)

---

## 📁 ไฟล์ที่สำคัญ

```
xauusd_rl_project/
├── trading_env.py              # Custom Trading Environment (Gym)
├── train_rl_model.py           # สคริปต์หลักสำหรับฝึกโมเดล
├── test_model.py               # ทดสอบโมเดลที่ฝึกเสร็จแล้ว
├── requirements_rl.txt         # Dependencies สำหรับ RL
├── view_tensorboard.sh         # เปิด TensorBoard
│
├── processed_data/
│   └── XAUUSD_READY_TO_TRAIN.parquet  # ข้อมูลที่พร้อมฝึก
│
├── models/
│   └── xauusd_model_checkpoints/      # โมเดลที่บันทึกไว้
│
└── logs/
    ├── tensorboard/                    # TensorBoard logs
    ├── monitor_train/                  # Training monitor logs
    └── monitor_eval/                   # Evaluation monitor logs
```

---

## 🚀 การติดตั้ง

### 1. ติดตั้ง Dependencies

```bash
pip install -r requirements_rl.txt
```

หรือติดตั้งแยกทีละตัว:
```bash
pip install stable-baselines3 gymnasium torch tensorboard
```

### 2. ตรวจสอบข้อมูล

ให้แน่ใจว่ามีไฟล์ `XAUUSD_READY_TO_TRAIN.parquet` แล้ว:
```bash
ls processed_data/XAUUSD_READY_TO_TRAIN.parquet
```

ถ้ายังไม่มี ให้รัน:
```bash
python prepare_for_training.py
```

---

## 🎯 การฝึกโมเดล

### เริ่มการฝึก

```bash
python train_rl_model.py
```

### การตั้งค่าในโค้ด (TrainingConfig)

| Parameter | Default | คำอธิบาย |
|-----------|---------|----------|
| `TOTAL_TIMESTEPS` | 1,000,000 | จำนวนก้าวในการฝึก |
| `LEARNING_RATE` | 3e-4 | อัตราการเรียนรู้ |
| `CHECKPOINT_FREQ` | 100,000 | บันทึกโมเดลทุกกี่ก้าว |
| `INITIAL_BALANCE` | $10,000 | เงินต้นเริ่มต้น |
| `LOT_SIZE` | 0.01 | ขนาดการเทรด (micro lot) |

### ระหว่างการฝึก

โมเดลจะ:
- ✅ บันทึก checkpoint ทุก 100,000 steps
- ✅ ประเมินผลบน test set ทุก 50,000 steps
- ✅ บันทึก metrics ไปยัง TensorBoard
- ✅ แสดง progress bar

### หยุดการฝึก

กด `Ctrl+C` โมเดลจะบันทึกอัตโนมัติ

---

## 📊 ดูผลการฝึก (TensorBoard)

### เปิด TensorBoard

```bash
chmod +x view_tensorboard.sh
./view_tensorboard.sh
```

หรือ:
```bash
tensorboard --logdir logs/tensorboard
```

เปิดเบราว์เซอร์: http://localhost:6006

### Metrics ที่ต้องดู

| Metric | คำอธิบาย | เป้าหมาย |
|--------|----------|----------|
| `rollout/ep_rew_mean` | Reward เฉลี่ยต่อ episode | เพิ่มขึ้นเรื่อยๆ |
| `train/entropy_loss` | การสำรวจของ AI | ลดลงช้าๆ |
| `train/policy_loss` | การเรียนรู้ policy | มีเสถียรภาพ |
| `train/value_loss` | การประเมินค่า state | ลดลงเรื่อยๆ |

---

## 🧪 ทดสอบโมเดล

### ทดสอบโมเดลที่ฝึกเสร็จ

```bash
python test_model.py --model models/xauusd_model_checkpoints/xauusd_ppo_final.zip
```

### ทดสอบ checkpoint เฉพาะ

```bash
python test_model.py --model models/xauusd_model_checkpoints/xauusd_ppo_500000_steps.zip
```

### ตัวเลือกเพิ่มเติม

```bash
python test_model.py \
  --model models/xauusd_model_checkpoints/xauusd_ppo_final.zip \
  --data processed_data/XAUUSD_READY_TO_TRAIN.parquet \
  --episodes 20
```

### ผลลัพธ์

โปรแกรมจะแสดง:
- ✅ Profit ต่อ episode
- ✅ Win rate
- ✅ จำนวนการเทรด
- ✅ กราฟผลลัพธ์ (บันทึกเป็นไฟล์ PNG)

---

## 🎮 Environment Details

### Action Space (4 actions)

| Action | Value | คำอธิบาย |
|--------|-------|----------|
| Hold | 0 | ไม่ทำอะไร |
| Buy | 1 | เปิด Long position |
| Sell | 2 | เปิด Short position |
| Close | 3 | ปิด position ปัจจุบัน |

### Observation Space

1. **Market Features** (50+ features):
   - OHLC ทุก timeframe (M15, H1, H4)
   - Technical indicators (RSI, ATR, MACD)
   - Moving averages
   - Log returns

2. **Position Features** (4 features):
   - Has position (0 or 1)
   - Position direction (-1, 0, 1)
   - Normalized profit
   - Normalized holding time

### Reward Function

```python
reward = (profit / initial_balance) + (equity_change / initial_balance)
```

**หลักการ:**
- 🎯 เน้น **Net Profit** จากการปิดออเดอร์
- 📈 เน้น **Equity Change** จากออเดอร์ที่เปิดอยู่
- 🔓 ให้ AI มีอิสระสำรวจ (ไม่มี penalty ด้าน drawdown หรือจำนวนไม้)

---

## 📈 โมเดล PPO (Proximal Policy Optimization)

### ทำไมเลือก PPO?

- ✅ เสถียรและเรียนรู้ได้เร็ว
- ✅ เหมาะกับ continuous และ discrete actions
- ✅ มี exploration ที่ดี (entropy)
- ✅ เป็นที่นิยมใน trading RL

### Hyperparameters

```python
learning_rate = 3e-4          # ปรับได้ถ้าเรียนรู้ช้า
n_steps = 2048                # ขนาด buffer
batch_size = 64               # ขนาด mini-batch
n_epochs = 10                 # จำนวนรอบ update ต่อ batch
gamma = 0.99                  # discount factor
ent_coef = 0.01               # exploration bonus
```

---

## 🛠️ Troubleshooting

### ปัญหา: Import Error

```bash
pip install --upgrade stable-baselines3 gymnasium torch
```

### ปัญหา: Out of Memory

ลด `n_steps` หรือ `batch_size` ใน `TrainingConfig`

### ปัญหา: Training ช้า

- ตรวจสอบว่ามี GPU หรือไม่: `torch.cuda.is_available()`
- ลด `TOTAL_TIMESTEPS` สำหรับทดสอบ
- เพิ่ม `n_envs` (parallel environments)

### ปัญหา: Reward ไม่เพิ่ม

- ปรับ `learning_rate` (ลองทั้ง 1e-4 และ 1e-3)
- เพิ่ม `ent_coef` (ให้ explore มากขึ้น)
- ตรวจสอบ reward function ใน `trading_env.py`

---

## 📝 Next Steps

หลังจากฝึกโมเดลเบื้องต้นเสร็จแล้ว สามารถพัฒนาต่อได้:

1. **ปรับ Reward Function**
   - เพิ่ม penalty สำหรับ drawdown
   - เพิ่ม penalty สำหรับจำนวนไม้มากเกินไป
   - เพิ่ม bonus สำหรับ win rate สูง

2. **ปรับ Environment**
   - เพิ่มข้อมูล sentiment
   - เพิ่ม spread แบบ dynamic
   - เพิ่ม slippage แบบสมจริง

3. **ทดลอง Algorithms อื่น**
   - A2C (faster but less stable)
   - SAC (good for continuous action)
   - TD3 (twin delayed DDPG)

4. **Hyperparameter Tuning**
   - ใช้ Optuna สำหรับ auto-tuning
   - Grid search หรือ random search

---

## 📞 Support

หากมีปัญหาหรือข้อสงสัย:
1. ตรวจสอบ log ไฟล์: `training.log`
2. ดู TensorBoard เพื่อดู metrics
3. ลองรันด้วย `TOTAL_TIMESTEPS` น้อยๆ เพื่อทดสอบก่อน

---

**Good luck with your RL training! 🚀🤖📈**

