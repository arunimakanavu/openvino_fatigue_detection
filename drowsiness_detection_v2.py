import cv2
import numpy as np
from openvino.runtime import Core
from collections import deque

# ---------------- OpenVINO Init ----------------
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

face_compiled = core.compile_model(face_model, "CPU")
headpose_compiled = core.compile_model(headpose_model, "CPU")
landmark_compiled = core.compile_model(landmark_model, "CPU")
gaze_compiled = core.compile_model(gaze_model, "CPU")

print("Gaze inputs:", [inp.any_name for inp in gaze_compiled.inputs])
print("Gaze outputs:", [out.any_name for out in gaze_compiled.outputs])

face_out = face_compiled.output(0)
lm_out = landmark_compiled.output(0)
gaze_out = gaze_compiled.output(0)


#---------------Eye Cropping Helper-----------------
RIGHT_EYE_IDX = list(range(60, 68))
LEFT_EYE_IDX  = list(range(68, 76))


def crop_eye(points, indices, frame):
    xs = [points[i][0] for i in indices]
    ys = [points[i][1] for i in indices]

    x1, x2 = max(min(xs)-5,0), min(max(xs)+5, frame.shape[1])
    y1, y2 = max(min(ys)-5,0), min(max(ys)+5, frame.shape[0])

    eye = frame[y1:y2, x1:x2]
    if eye.size == 0:
        return None

    eye = cv2.resize(eye, (60,60))
    eye = eye.astype(np.float32) / 255.0
    eye = eye.transpose(2,0,1)[np.newaxis, :]
    return eye


# ---------------- Temporal Buffers ---------------- 
pitch_history = deque(maxlen=15)
gaze_history = deque(maxlen=15)
flag = 0
alert_active = False

PITCH_THRESHOLD = 20.0
PITCH_VAR_THRESHOLD = 5.0
FRAME_CHECK = 15

# ---------------- Video ----------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # ---------- Face Detection ----------
    fd_input = cv2.resize(frame, (672, 384))
    fd_input = fd_input.transpose(2, 0, 1)[np.newaxis, :]

    detections = face_compiled([fd_input])[face_out][0][0]

    best_det = None
    best_conf = 0.0

    for det in detections:
        conf = det[2]
        if conf > best_conf:
            best_conf = conf
            best_det = det

    if best_det is None or best_conf < 0.5:
        cv2.imshow("Head Pose Drowsiness", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    xmin = int(best_det[3] * w)
    ymin = int(best_det[4] * h)
    xmax = int(best_det[5] * w)
    ymax = int(best_det[6] * h)

    face_roi = frame[ymin:ymax, xmin:xmax]
    if face_roi.size == 0:
        continue


    # ---------- Head Pose ----------
    hp_input = cv2.resize(face_roi, (60, 60))
    hp_input = hp_input.transpose(2, 0, 1)[np.newaxis, :]

    hp_result = headpose_compiled([hp_input])

    yaw   = hp_result[headpose_compiled.output("angle_y_fc")][0][0]
    pitch = hp_result[headpose_compiled.output("angle_p_fc")][0][0]
    roll  = hp_result[headpose_compiled.output("angle_r_fc")][0][0]

    pitch_history.append(pitch)

    if len(pitch_history) < 3:
        avg_pitch = pitch
        pitch_var = 999.0   # treat as unstable initially
    else:
        avg_pitch = sum(pitch_history) / len(pitch_history)
        pitch_var = np.var(pitch_history)
        
    head_drowsy = (
        avg_pitch > PITCH_THRESHOLD and
        pitch_var < PITCH_VAR_THRESHOLD
    )

        
    # ---- Facial landmarks ----
    lm_input = cv2.resize(face_roi, (64,64))
    lm_input = lm_input.transpose(2,0,1)[np.newaxis, :]

    lm_result = landmark_compiled([lm_input])[lm_out].reshape(-1)

    # Convert to image coordinates
    fh, fw = face_roi.shape[:2]
    points = []
    for i in range(98):
        x = int(lm_result[2*i] * fw) + xmin
        y = int(lm_result[2*i+1] * fh) + ymin
        points.append((x,y))

    left_eye_img  = crop_eye(points, LEFT_EYE_IDX, frame)
    right_eye_img = crop_eye(points, RIGHT_EYE_IDX, frame)

    if left_eye_img is None or right_eye_img is None:
        print("!!! Eye crop failed")
        continue
        
        
    #----------- Gaze Inference ------------

    gaze_inputs = {
        "left_eye_image": left_eye_img,
        "right_eye_image": right_eye_img,
        "head_pose_angles": np.array([[yaw, pitch, roll]], dtype=np.float32)
    }

    gaze_vec = gaze_compiled(gaze_inputs)[gaze_out][0]
    gx, gy, gz = gaze_vec

    gaze_history.append(gy)
    if len(gaze_history) < 5:
        gaze_stable = False
    else:
        gaze_var = np.var(gaze_history)
        gaze_stable = gaze_var < 0.0008
    
    print(f"Gaze vector: gx={gx:.3f}, gy={gy:.3f}, gz={gz:.3f}")

    # ---------- Drowsiness Logic ----------

    if head_drowsy and gaze_stable:
        flag += 1
    else:
        flag = 0

    if flag >= FRAME_CHECK:
        alert_active = True
    else:
        alert_active = False


    # ---------- Visualization ----------
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    cv2.putText(frame,
            f"Pitch: {avg_pitch:.1f}  GazeY: {gy:.2f}",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2)
            
    cv2.putText(frame,
            f"GazeVar: {np.var(gaze_history):.5f}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2)

            
    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2

    cv2.arrowedLine(
        frame,
        (cx, cy),
        (int(cx + gx * 100), int(cy - gy * 100)),
        (0, 255, 255),
        2
    )

                    
    if alert_active:
        cv2.putText(frame,
                    "DROWSINESS ALERT",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3)


    cv2.imshow("Head Pose Drowsiness", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
