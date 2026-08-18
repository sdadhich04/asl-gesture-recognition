# TinyML ASL Gesture Recognition — Arduino Nano 33 BLE Sense

**EE 446: Tiny Machine Learning for Ultra Low-Power Edge Computing | University of Washington, Spring 2026**

Classifying hand gestures ("hi" and "sup") in real time using the onboard IMU on an **Arduino Nano 33 BLE Sense**, with a TensorFlow Lite Micro model running entirely on the microcontroller.

---

## What it does

The system detects significant motion via the accelerometer, then captures 119 samples of 6-axis IMU data (3-axis acceleration + 3-axis gyroscope) and classifies the motion as one of two gestures:

| Gesture | Description |
|---------|-------------|
| **hi** | Wave gesture |
| **sup** | Upward nod/raise gesture |

---

## Pipeline overview

1. **Data collection** — IMU data recorded as CSV files (`hi.csv`, `sup.csv`) using the Nano 33 BLE Sense
2. **Model training** — Dense neural network trained on 119-sample × 6-feature windows in TensorFlow
3. **TFLite conversion** — Model converted to TensorFlow Lite and quantized for microcontroller deployment
4. **Model compression** — TFLite Micro model compressed and exported as `model.h` C array
5. **Arduino deployment** — Sketch loads `model.h`, waits for motion trigger, runs inference, prints results over Serial

---

## Model

- **Input:** 119 samples × 6 features (aX, aY, aZ normalized to [0,1]; gX, gY, gZ normalized to [0,1])
- **Architecture:** Dense neural network (TFLite Micro)
- **Output:** 2-class softmax (hi, sup)
- **Tensor arena:** 16 KB
- **Supports:** Arduino Nano 33 BLE Sense Rev1 (LSM9DS1 IMU) and Rev2 (BMI270/BMM150 IMU)

---

## Repository contents

```
EE446_TinyML_Lab9.ipynb              ← Full pipeline: data loading → training → TFLite conversion → compression
TinyML-Lab9.pdf                      ← Lab instructions
data/
  hi.csv                             ← IMU recordings for "hi" gesture
  sup.csv                            ← IMU recordings for "sup" gesture
model/
  gesture_model.tflite               ← Trained TFLite model (148 KB)
  model.h                            ← Model as C array for Arduino (915 KB)
arduino/
  lab9-classifier-dual-board.ino     ← Arduino sketch (supports Rev1 and Rev2 boards)
```

---

## Hardware setup

- **Arduino Nano 33 BLE Sense** (Rev1 or Rev2)
- Set `#define USE_NANO_33_BLE_REV2_IMU 1` in the sketch for Rev2, `0` for Rev1
- Upload via Arduino IDE with `TensorFlowLite` library installed
- Open Serial Monitor at 9600 baud — gesture probabilities print after each detected motion

---

## Dependencies

**Python (notebook):** `numpy`, `pandas`, `matplotlib`, `tensorflow`, `scikit-learn`

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn
```

**Arduino:** `TensorFlowLite` library, `Arduino_BMI270_BMM150` (Rev2) or `Arduino_LSM9DS1` (Rev1)

---

## Authors

Sparsh Dadhich — University of Washington, ECE / Neuroscience
