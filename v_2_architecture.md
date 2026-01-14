# Drowsiness Detection – Version 2 (Gaze Estimation + Head Pose)

## 1. Motivation for Version 2

Version 1 established that eyelid-based heuristics (EAR-style logic) are not reliable when driven directly by OpenVINO’s 98-point facial landmark model. The primary limitation was landmark instability for fine eyelid motion, which the model was not designed to solve.

Version 2 shifts the design from hand-crafted geometric heuristics to semantic signals produced by trained neural networks, aligning with OpenVINO’s intended usage patterns and official demos.

---

## 2. Design Philosophy

**Key principle**:

> Use each OpenVINO model strictly for the task it was trained for, and perform drowsiness reasoning at the *temporal aggregation layer*, not at the raw pixel or landmark level.

This version focuses on behavioral cues associated with drowsiness:
- Prolonged downward gaze
- Sustained head droop (pitch)
- Reduced gaze stability over time

---

## 3. Model Stack

### 3.1 Face Detection
- **Model**: `face-detection-adas-0001`
- **Role**:
  - Detect primary face per frame
  - Provide face ROI for downstream models
- **Notes**:
  - Highest-confidence face selected per frame
  - No temporal logic applied at this stage

---

### 3.2 Facial Landmarks (Utility Role)
- **Model**: `facial-landmarks-98-detection-0001`
- **Role**:
  - Identify eye regions
  - Support precise cropping of left and right eye images
- **Explicit Non-Goals**:
  - Not used for eyelid distance
  - Not used for blink / EAR computation

This is consistent with OpenVINO’s official gaze estimation demo.

---

### 3.3 Head Pose Estimation
- **Model**: `head-pose-estimation-adas-0001`
- **Input**:
  - Face ROI
- **Output**:
  - Yaw (left/right)
  - Pitch (up/down)
  - Roll (tilt)

#### Relevance to Drowsiness
- **Pitch** is a strong indicator of fatigue:
  - Sustained downward pitch → head droop
  - Repeated nodding → micro-sleep behavior
- Head pose is a **stable signal** compared to landmarks

---

### 3.4 Gaze Estimation
- **Model**: `gaze-estimation-adas-0002`
- **Inputs**:
  - Left eye image
  - Right eye image
  - Head pose angles
- **Output**:
  - Gaze direction vector (gx, gy, gz)

#### Relevance to Drowsiness
- Downward or unstable gaze correlates with fatigue
- Eye closure causes degraded or collapsed gaze vectors
- Provides a learned eye-behavior signal, not a geometric approximation

---

## 4. End-to-End Pipeline

```
Camera Frame
   ↓
Face Detection
   ↓
Face ROI
   ├── Head Pose Estimation
   │       ↓
   │    (yaw, pitch, roll)
   │
   └── Facial Landmarks
           ↓
       Eye Cropping
           ↓
       Gaze Estimation
           ↓
   (gaze vector)
           ↓
Temporal Aggregation
           ↓
Drowsiness Decision Logic
           ↓
Alert / Visualization
```

---

## 5. Temporal Aggregation Strategy

All behavioral reasoning is performed across time.

### Signals Aggregated
- Head pitch values
- Gaze vector magnitude and direction
- Gaze stability (variance)

### Techniques Used
- Sliding window buffers (`deque`)
- Median / mean smoothing
- Frame counters to suppress transient events (blinks)

This ensures robustness against:
- single-frame noise
- lighting variation
- momentary head movements

---

## 6. Drowsiness Decision Logic (Rule-Based)

Version 2 uses simple, explainable heuristics:

### Example Conditions
- Head pitch exceeds threshold for N consecutive frames
- Gaze vector indicates downward gaze for sustained duration
- Gaze instability over time crosses tolerance

### Combined Rule (Conceptual)

```
IF (head_pitch_droop) AND (interrupted_gaze)
THEN drowsy
```

This avoids dependence on fragile absolute thresholds and instead relies on semantic consistency over time.

---

## 7. Key Advantages over Version 1

| Aspect | Version 1 | Version 2 |
|------|----------|-----------|
| Eye modeling | Hand-crafted geometry | Trained gaze model |
| Head behavior | Not used | Explicitly modeled |
| Landmark reliance | Heavy | Minimal (cropping only) |
| Temporal reasoning | Partial | Core design principle |
| Stability | Low | High |

---

## 8. Implementation Strategy

### Phase 1
- Integrate all models and verify inference outputs
- Visualize:
  - face box
  - head pose angles
  - gaze vector direction

### Phase 2
- Implement temporal buffers
- Tune thresholds empirically

### Phase 3
- Add alerting and demo UI
- Document behavior under different conditions

---

## 9. Limitations (Known & Accepted)

- Not a medical-grade fatigue detector
- Rule-based logic, not end-to-end learning
- Performance depends on camera quality and face visibility

These are acceptable for a **demo and architectural prototype**.

---

## 10. Conclusion

Version 2 represents a **structural correction**, not an incremental tweak.

By shifting from raw geometric heuristics to **semantic model outputs + temporal logic**, the system becomes:
- more stable
- easier to explain
- aligned with OpenVINO’s reference implementations

This version provides a solid foundation for further refinement or ML-based fusion in future iterations.

---

**Status**: Active architecture
**Supersedes**: Version 1 (Face + Landmark-based EAR)

