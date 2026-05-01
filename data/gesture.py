import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components import containers
import pyautogui
import numpy as np
import time

# ── Settings ──────────────────────────────────────────────
SMOOTHING    = 10     # Higher = smoother but slightly slower
CLICK_DIST   = 25     # Pinch distance (pixels) to trigger click
SCROLL_SENS  = 3      # Scroll speed
MODEL_PATH   = "hand_landmarker.task"
# ──────────────────────────────────────────────────────────

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

scr_w, scr_h = pyautogui.size()

# ── Build the hand landmarker ──────────────────────────────
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
    running_mode=vision.RunningMode.VIDEO
)
landmarker = vision.HandLandmarker.create_from_options(options)

# ── Helpers ───────────────────────────────────────────────
def dist(p1, p2):
    return np.hypot(p1[0]-p2[0], p1[1]-p2[1])

def smooth(curr, prev, f):
    return prev + (curr - prev) / f

# ── State ─────────────────────────────────────────────────
prev_x, prev_y   = 0, 0
click_cd         = 0
rclick_cd        = 0
scroll_cd        = 0
frame_ts         = 0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Gesture Mouse running — press Q to quit")
print("Gestures:  INDEX=move  |  INDEX+THUMB pinch=click  |  MIDDLE+THUMB pinch=scroll")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    # Convert to MediaPipe image
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    frame_ts += 33   # ~30fps timestamp in ms
    result = landmarker.detect_for_video(mp_image, frame_ts)

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]

        # Pixel coords helper
        def px(i):
            return int(lm[i].x * w), int(lm[i].y * h)

        index_tip  = px(8)
        middle_tip = px(12)
        ring_tip   = px(16)
        thumb_tip  = px(4)
        wrist      = px(0)

        # ── MOVE — index finger controls cursor ────────────
        cursor_x = np.interp(lm[8].x, [0.1, 0.9], [0, scr_w])
        cursor_y = np.interp(lm[8].y, [0.1, 0.9], [0, scr_h])

        sx = smooth(cursor_x, prev_x, SMOOTHING)
        sy = smooth(cursor_y, prev_y, SMOOTHING)
        prev_x, prev_y = sx, sy
        pyautogui.moveTo(sx, sy, duration=0)

        # Draw landmarks
        for i in range(21):
            pt = px(i)
            cv2.circle(frame, pt, 4, (180, 180, 180), -1)

        # Highlight fingertips
        cv2.circle(frame, index_tip,  12, (0, 255, 100),  -1)
        cv2.circle(frame, thumb_tip,  10, (255, 200, 0),  -1)
        cv2.circle(frame, middle_tip, 10, (100, 180, 255), -1)

        # ── LEFT CLICK — index + thumb pinch ───────────────
        pinch = dist(index_tip, thumb_tip)
        if pinch < CLICK_DIST and click_cd == 0:
            pyautogui.click()
            click_cd = 25
            cv2.putText(frame, "LEFT CLICK", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 200), 3)

        # ── RIGHT CLICK — middle + thumb pinch ─────────────
        rpinch = dist(middle_tip, thumb_tip)
        if rpinch < CLICK_DIST and rclick_cd == 0 and pinch >= CLICK_DIST:
            pyautogui.rightClick()
            rclick_cd = 25
            cv2.putText(frame, "RIGHT CLICK", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 150, 255), 3)

        # ── SCROLL — ring + thumb pinch, move hand up/down ─
        scpinch = dist(ring_tip, thumb_tip)
        if scpinch < CLICK_DIST and scroll_cd == 0:
            vert = wrist[1] - ring_tip[1]
            if vert > 30:
                pyautogui.scroll(SCROLL_SENS)
                cv2.putText(frame, "SCROLL UP", (20, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 255, 0), 2)
            elif vert < -10:
                pyautogui.scroll(-SCROLL_SENS)
                cv2.putText(frame, "SCROLL DOWN", (20, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 120, 0), 2)
            scroll_cd = 6

        # Cooldowns
        if click_cd  > 0: click_cd  -= 1
        if rclick_cd > 0: rclick_cd -= 1
        if scroll_cd > 0: scroll_cd -= 1

    # ── HUD ───────────────────────────────────────────────
    cv2.rectangle(frame, (0,0), (w, 38), (20,20,20), -1)
    cv2.putText(frame,
        "INDEX=move | PINCH(idx+thumb)=click | PINCH(mid+thumb)=rclick | PINCH(ring+thumb)=scroll",
        (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 255, 160), 1)

    cv2.imshow("Gesture Mouse  [Q = quit]", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()