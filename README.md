# Drowsiness Detection (OpenVINO – Head Pose + Gaze)

This repository contains a **minimal, working proof-of-concept** for driver drowsiness detection using **OpenVINO**. 
The system uses **head pose estimation as the primary signal** and **gaze stability as a secondary confirmation signal**.

> ⚠️ This is a demo / POC implementation intended for experimentation and understanding model behavior, not a production-ready safety system.

---

## 1. What This Demo Does

- Captures live video from a webcam
- Detects the face
- Estimates head pose (yaw, pitch, roll)
- Estimates gaze direction from eye crops
- Triggers a **DROWSINESS ALERT** when:
  - Head pitch indicates sustained downward posture **and**
  - Gaze becomes abnormally stable over time

---

## 2. Prerequisites

### System
- Ubuntu 20.04 / 22.04 (recommended)
- Python 3.9 – 3.12
- Webcam

### Python Dependencies
Create a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Typical `requirements.txt`:

```text
openvino>=2023.3
opencv-python
numpy
```

---

## 3. Required OpenVINO Models

**Model conversion is NOT required**. Pre-converted IR models from Open Model Zoo are used.

Download the following models using Open Model Zoo tools:

```bash
omz_downloader --name face-detection-adas-0001
omz_downloader --name facial-landmarks-98-detection-0001
omz_downloader --name head-pose-estimation-adas-0001
omz_downloader --name gaze-estimation-adas-0002
```

Ensure the models are available under:

```text
models/
├── face-detection-adas-0001/FP16/
├── facial-landmarks-98-detection-0001/FP16/
├── head-pose-estimation-adas-0001/FP16/
└── gaze-estimation-adas-0002/FP16/
```

> If FP16 is unavailable, FP32 can also be used (update paths in code accordingly).

---

## 4. Running the Demo

From the project root:

```bash
python3 fatigue_detection_demo.py
```

Press **`q`** to quit.

---

## 5. Output Explanation

### On-screen
- **Face bounding box**
- **Gaze arrow** (yellow): visualized gaze direction
- **Pitch & gaze metrics** (debug overlay)
- **DROWSINESS ALERT** text when conditions are met


## 6. Alert Logic (High-level)

- **Head Pose**: primary indicator (sustained downward pitch)
- **Gaze Stability**: secondary confirmation (low variance over time)
- Alert is raised only when **both conditions persist for N frames** (in this case 15 frames)

This fusion strategy reduces false positives caused by normal downward gaze (e.g., reading, looking at dashboard).

---

## 7. Known Limitations

- Sensitive to camera placement and lighting
- Gaze estimation is subtle and user-dependent
- No eye-closure (EAR) model used in this version
- Single-face assumption

---

## 8. Conclusion

This POC demonstrates how **head pose and gaze estimation can be fused** using OpenVINO models to detect potential drowsiness scenarios in real time. The project focuses on clarity, debuggability, and correctness of model usage rather than completeness.


