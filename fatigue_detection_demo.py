import cv2
import numpy as np
from openvino.runtime import Core
from collections import deque

# ============================================================
# 1. MODEL INITIALIZATION
# ============================================================

core = Core()

face_model = core.read_model(
    "models/face-detection-adas-0001/FP16/face-detection-adas-0001.xml"
)
headpose_model = core.read_model(
    "models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml"
)
landmark_model = core.read_model(
    "models/facial-landmarks-98-detection-0001/FP16/facial-landmarks-98-detection-0001.xml"
)
gaze_model = core.read_model(
    "models/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml"
)

face_compiled = core.compile_model(face_model, "AUTO")
headpose_compiled = core.compile_model(headpose_model, "AUTO")
landmark_compiled = core.compile_model(landmark_model, "AUTO")
gaze_compiled = core.compile_model(gaze_model, "GPU")

face_out = face_compiled.output(0)
lm_out = landmark_compiled.output(0)
gaze_out = gaze_compiled.output(0)

# ============================================================
# 2. CONSTANTS & STATE
# ============================================================

RIGHT_EYE_IDX = list(range(60, 68))
LEFT_EYE_IDX  = list(range(68, 76))

PITCH_THRESHOLD = 20.0
PITCH_VAR_THRESHOLD = 5.0
GAZE_VAR_THRESHOLD = 0.0008
FRAME_CHECK = 15

pitch_history = deque(maxlen=15)
gaze_history = deque(maxlen=15)
alert_counter = 0

# ============================================================
# 3. HELPER FUNCTIONS (stateless utilities)
# ============================================================

def crop_eye(points, indices, frame):
    xs = [points[i][0] for i in indices]
    ys = [points[i][1] for i in indices]

    x1, x2 = max(min(xs) - 5, 0), min(max(xs) + 5, frame.shape[1])
    y1, y2 = max(min(ys) - 5, 0), min(max(ys) + 5, frame.shape[0])

    eye = frame[y1:y2, x1:x2]
    if eye.size == 0:
        return None

    eye = cv2.resize(eye, (60, 60))
    eye = eye.astype(np.float32) / 255.0
    eye = eye.transpose(2, 0, 1)[np.newaxis, :]
    return eye

# ============================================================
# 4. PER-FRAME INFERENCE 
# ============================================================

def run_inference(frame):
    h, w = frame.shape[:2]

    # Face detection
    fd_input = cv2.resize(frame, (672, 384))
    fd_input = fd_input.transpose(2, 0, 1)[np.newaxis, :]
    detections = face_compiled([fd_input])[face_out][0][0]

    best_det, best_conf = None, 0.0
    for det in detections:
        if det[2] > best_conf:
            best_conf, best_det = det[2], det

    if best_det is None or best_conf < 0.5:
        return None

    xmin = int(best_det[3] * w)
    ymin = int(best_det[4] * h)
    xmax = int(best_det[5] * w)
    ymax = int(best_det[6] * h)

    face_roi = frame[ymin:ymax, xmin:xmax]
    if face_roi.size == 0:
        return None

    # Head pose
    hp_input = cv2.resize(face_roi, (60, 60))
    hp_input = hp_input.transpose(2, 0, 1)[np.newaxis, :]
    hp_result = headpose_compiled([hp_input])

    yaw   = hp_result[headpose_compiled.output("angle_y_fc")][0][0]
    pitch = hp_result[headpose_compiled.output("angle_p_fc")][0][0]
    roll  = hp_result[headpose_compiled.output("angle_r_fc")][0][0]

    # Landmarks
    lm_input = cv2.resize(face_roi, (64, 64))
    lm_input = lm_input.transpose(2, 0, 1)[np.newaxis, :]
    lm_result = landmark_compiled([lm_input])[lm_out].reshape(-1)

    fh, fw = face_roi.shape[:2]
    points = [
        (int(lm_result[2*i] * fw) + xmin,
         int(lm_result[2*i+1] * fh) + ymin)
        for i in range(98)
    ]

    left_eye = crop_eye(points, LEFT_EYE_IDX, frame)
    right_eye = crop_eye(points, RIGHT_EYE_IDX, frame)
    if left_eye is None or right_eye is None:
        return None

    # Gaze
    gaze_inputs = {
        "left_eye_image": left_eye,
        "right_eye_image": right_eye,
        "head_pose_angles": np.array([[yaw, pitch, roll]], dtype=np.float32)
    }

    gaze_vec = gaze_compiled(gaze_inputs)[gaze_out][0]

    return {
        "bbox": (xmin, ymin, xmax, ymax),
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
        "gaze": gaze_vec
    }

# ============================================================
# 5. TEMPORAL AGGREGATION
# ============================================================

def update_temporal_state(pitch, gaze_y):
    pitch_history.append(pitch)
    gaze_history.append(gaze_y)

    avg_pitch = np.mean(pitch_history)
    pitch_var = np.var(pitch_history) if len(pitch_history) > 2 else 999.0
    gaze_var  = np.var(gaze_history) if len(gaze_history) > 4 else 999.0

    return avg_pitch, pitch_var, gaze_var

# ============================================================
# 6. USE-CASE DECISION LOGIC
# ============================================================

def evaluate_drowsiness(avg_pitch, pitch_var, gaze_var):
    head_stable = avg_pitch > PITCH_THRESHOLD and pitch_var < PITCH_VAR_THRESHOLD
    gaze_stable = gaze_var < GAZE_VAR_THRESHOLD
    return head_stable and gaze_stable

# ============================================================
# 7. MAIN LOOP
# ============================================================

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = run_inference(frame)
    if result is None:
        cv2.imshow("Head Pose Drowsiness", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    xmin, ymin, xmax, ymax = result["bbox"]
    gx, gy, gz = result["gaze"]

    avg_pitch, pitch_var, gaze_var = update_temporal_state(
        result["pitch"], gy
    )

    drowsy = evaluate_drowsiness(avg_pitch, pitch_var, gaze_var)

    if drowsy:
        alert_counter += 1
    else:
        alert_counter = 0

    alert_active = alert_counter >= FRAME_CHECK

    # ========================================================
    # 8. VISUALIZATION
    # ========================================================

    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    cv2.putText(frame, f"Pitch: {avg_pitch:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"GazeVar: {gaze_var:.5f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
    cv2.arrowedLine(
        frame, (cx, cy),
        (int(cx + gx * 100), int(cy - gy * 100)),
        (0, 255, 255), 2
    )

    if alert_active:
        cv2.putText(frame, "DROWSINESS ALERT", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.imshow("Head Pose Drowsiness", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

