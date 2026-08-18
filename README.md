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
4. **Model export** — TFLite model exported as `model.h` C array for Arduino
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
EE446_TinyML_Lab9.ipynb              ← Full pipeline: data loading → training → TFLite conversion → export
TinyML-Lab9.pdf                      ← Lab instructions
data/
  hi.csv                             ← IMU recordings for "hi" gesture
  sup.csv                            ← IMU recordings for "sup" gesture
model/
  gesture_model.tflite               ← Trained TFLite model (148 KB)
arduino/
  lab9-classifier-dual-board.ino     ← Arduino sketch (supports Rev1 and Rev2 boards)
  model.h                            ← Model as C array (must be in same folder as sketch)
```

---

## Quick start

### Run the notebook (train / retrain)

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn
jupyter notebook EE446_TinyML_Lab9.ipynb
```

The notebook loads `data/hi.csv` and `data/sup.csv`, trains a dense network, converts to TFLite, and saves the output as `model.h`. Copy the generated `model.h` into the `arduino/` folder before uploading to the board.

### Flash the Arduino sketch

1. Install Arduino libraries: `TensorFlowLite`, plus `Arduino_BMI270_BMM150` (Rev2) or `Arduino_LSM9DS1` (Rev1)
2. Open `arduino/lab9-classifier-dual-board.ino` in Arduino IDE
3. Set the board flag at the top of the sketch:
   ```cpp
   #define USE_NANO_33_BLE_REV2_IMU 1  // 1 = Rev2 (BMI270), 0 = Rev1 (LSM9DS1)
   ```
4. Select **Arduino Nano 33 BLE** (or BLE Sense) as the board, upload
5. Open **Serial Monitor at 9600 baud** — after each detected motion, the sketch prints softmax probabilities for each gesture

> **Note:** `arduino/model.h` is required to compile the sketch. If you regenerate the model in the notebook, copy the new `model.h` into the `arduino/` folder before re-uploading.

---

## Hardware

- **Arduino Nano 33 BLE Sense** (Nordic nRF52840, 256 KB flash, 64 KB RAM, onboard IMU)

---

## Authors

Sparsh Dadhich — University of Washington, ECE / Neuroscience
