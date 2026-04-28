# V12 Genesis — ไฟล์โมดูลและบทบาท

โปรเจกต์ **V12 Genesis** (Physics-based RL สำหรับ XAUUSD / WeTrade) รวมไฟล์หลักดังนี้:

| ไฟล์ | บทบาท |
|------|--------|
| `v12_config.py` | พารามิเตอร์รวม: broker, physics, Hurst, Z-score, Monte Carlo, risk, reward, adversarial, grid, sim |
| `v12_monte_carlo.py` | จำลอง GBM แบบ vectorized (`run_mc_summary` → confidence, margin_at_risk, path_var_95) |
| `v12_physics_env.py` | Gymnasium `V12PhysicsEnv`: Stay/Buy/Sell, mask margin & Anti-Doi, MC, hedge/Houdini/adversarial |
| `v12_brain_ppo.py` | เทรน **MaskablePPO** + **LSTM** + `VecFrameStack` + `ActionMasker` |
| `v12_executor.py` | Export **ONNX** + CSV bridge สำหรับ MT5 |
| `v12_eval.py` | ประเมินโมเดล .zip บนชุด **test/train** (reward, margin, equity) |
| `requirements-v12.txt` | แพ็กเกจ Python ที่ใช้ |

## รันที่ใช้บ่อย

```bash
cd rl_xauusd
pip install -r requirements-v12.txt
# macOS + PyTorch บางเวอร์ชัน: ใส่ก่อนรันถ้าเจอ libomp ซ้ำ
export KMP_DUPLICATE_LIB_OK=TRUE

# เทรน (ต้องมีไฟล์ OHLCV เช่น XAUUSD_M5_wetrade.csv)
python v12_brain_ppo.py --steps 500000 --data XAUUSD_M5_wetrade.csv

# ประเมิน out-of-sample (ต้องมีไฟล์โมเดล .zip)
python v12_eval.py --model v12_maskable_ppo_lstm.zip --split test --episodes 5 --max-steps 30000

# เทรนรอบล่าสุด (ดู `v12_config` RewardSpec + `--ent-coef` ใน `v12_brain_ppo`)
# ตัวอย่างรอบ v4 (หลังสรุป run1–run3): ent กลางๆ + reward สมดุลใน config แล้ว
python v12_brain_ppo.py --steps 800000 --save v12_run4 --ent-coef 0.01

# Export ONNX หลังมีโมเดล .zip
python v12_executor.py export --model v12_maskable_ppo_lstm.zip --onnx v12_policy.onnx --spec v12_onnx_spec.json

# ทดสอบ bridge ตัวอย่าง
python v12_executor.py bridge-demo --dir .
```

## สมมติฐานสำคัญ

- Action RL เฉพาะ **Stay / Buy / Sell** (ไม่มี Close จาก policy); ปิดบางส่วนจากระบบ Houdini / hedge เป็นคนละชั้นกับ RL
- **Margin &lt; 100%** → mask เปิด Buy/Sell (hard)
- ขนาด obs ต่อ step = `n_obs_features` (default 64); เทรนใช้ stack **8** เฟรม → เวกเตอร์เข้าโมเดล **512** มิติ
