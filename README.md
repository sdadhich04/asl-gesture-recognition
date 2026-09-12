# ASL Gesture Recognition on Arduino

This project trains and deploys a small TensorFlow Lite Micro classifier for two hand gestures, `hi` and `sup`, using accelerometer and gyroscope data from an Arduino Nano 33 BLE / BLE Sense IMU.

## What It Does

- Reads six IMU channels: `aX`, `aY`, `aZ`, `gX`, `gY`, and `gZ`.
- Uses 119 samples per gesture recording.
- Normalizes acceleration with `(value + 4.0) / 8.0` and gyroscope values with `(value + 2000.0) / 4000.0`.
- Trains a two-class Keras model and converts it to TensorFlow Lite.
- Runs inference on Arduino after an acceleration threshold detects motion.
- Prints softmax scores for `hi` and `sup` over Serial at 9600 baud.

## Model

The notebook builds this Keras model:

```text
Dense(50, activation="relu")
Dense(15, activation="relu")
Dense(2, activation="softmax")
```

It compiles the model with `rmsprop`, `categorical_crossentropy`, and `accuracy`, trains for 100 epochs with batch size 16, and converts the Keras model with `tf.lite.TFLiteConverter.from_keras_model(model)`. The checked-in TensorFlow Lite model is `model/gesture_model.tflite`.

## Hardware and Tools

- Arduino Nano 33 BLE / BLE Sense
- Rev2 IMU library: `Arduino_BMI270_BMM150`
- Original Rev1 IMU library: `Arduino_LSM9DS1`
- TensorFlow Lite Micro Arduino library
- Python notebook using `numpy`, `pandas`, `matplotlib`, and `tensorflow`

The Arduino sketch defaults to Rev2 with:

```cpp
#define USE_NANO_33_BLE_REV2_IMU 1
```

Set that flag to `0` for the original Rev1 IMU library path.

## Files

```text
EE446_TinyML_Lab9.ipynb
  Notebook for loading gesture CSVs, training the model, converting to TFLite,
  and exporting a C header.

TinyML-Lab9.pdf
  Lab handout included in the repository.

data/hi.csv
data/sup.csv
  Gesture data files. Each has six IMU columns and 5,950 data rows, which the
  notebook parses as 50 recordings of 119 samples.

model/gesture_model.tflite
model/model.h
  Converted TensorFlow Lite model and exported C array.

arduino/lab9-classifier-dual-board.ino
arduino/model.h
  Arduino classifier sketch and the model header used by the sketch.
```

## How To Run

To retrain or regenerate the model header:

```bash
pip install numpy pandas matplotlib tensorflow
jupyter notebook EE446_TinyML_Lab9.ipynb
```

The notebook writes generated outputs under `build/`. Copy the regenerated `build/model.h` to `arduino/model.h` before uploading the Arduino sketch.

To run on Arduino:

1. Install the TensorFlow Lite Micro Arduino library.
2. Install `Arduino_BMI270_BMM150` for Rev2 boards, or `Arduino_LSM9DS1` for original Rev1 boards.
3. Open `arduino/lab9-classifier-dual-board.ino` in the Arduino IDE.
4. Confirm the Rev1/Rev2 flag at the top of the sketch.
5. Upload to the Arduino Nano 33 BLE / BLE Sense.
6. Open Serial Monitor at 9600 baud.

## Project Context

This reads primarily as an EE 446 TinyML Lab 9 coursework project that has been cleaned up as a standalone GitHub repository. The notebook title, included `TinyML-Lab9.pdf`, Arduino sketch name, and initial commit all reference Lab 9 / EE 446. Later commits fix repository structure, adjust hardware support, and add a license.

## Credits

- Author: Sparsh Dadhich
- Course context shown in the notebook: EE446 TinyML Lab 9, University of Washington
- License: MIT, see `LICENSE`
