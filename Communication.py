import cv2
import numpy as np
import requests
import threading
import time
import winsound
from datetime import datetime

# ==========================================
# 1. CẤU HÌNH BLYNK
# ==========================================
BLYNK_AUTH_TOKEN = "g9Hr0MuhXc1HigvnA57IW0ODmVa5Rzs1" 
# Đã sửa 'api2' thành 'api' để Blynk nhận được request
BLYNK_BASE_URL = "https://sgp1.blynk.cloud/external/api"

SOURCE = 0
fire_counter = 0
warmup_frames = 0
CONFIRM_FRAMES = 25     
ALERT_INTERVAL = 30      
last_alert_time = 0      
last_state = 0           

top_left = (100, 100)
bottom_right = (540, 380)

# ==========================================
# 2. HÀM TRUYỀN TIN
# ==========================================

def trigger_alarm_process(frame, msg_type, is_repeat=False):
    def task():
        global last_alert_time
        try:
            start_time = time.perf_counter()
            time_now = datetime.now().strftime("%H:%M:%S")
            print(f"\n--- [ALERT] {msg_type} LÚC: {time_now} ---")
            
            # Gửi Event lên Blynk
            requests.get(f"{BLYNK_BASE_URL}/logEvent?token={BLYNK_AUTH_TOKEN}&code=fire_alert", timeout=5)

            # Cập nhật V1 và V2
            status_msg = f"CANH_BAO_{msg_type}_{time_now}"
            requests.get(f"{BLYNK_BASE_URL}/update?token={BLYNK_AUTH_TOKEN}&v1=1", timeout=5)
            requests.get(f"{BLYNK_BASE_URL}/update?token={BLYNK_AUTH_TOKEN}&v2={status_msg}", timeout=5)
            
            latency = (time.perf_counter() - start_time) * 1000 
            print(f"[OK] Đã báo {msg_type} về server. Latency: {latency:.2f} ms")
            
            if not is_repeat:
                cv2.imwrite(f"BANG_CHUNG_{datetime.now().strftime('%H%M%S')}.jpg", frame)
            for _ in range(3): winsound.Beep(2500, 800) 
        except Exception as e: print(f"[ERROR]: {e}")
    threading.Thread(target=task).start()

def reset_system_status():
    def task():
        try:
            requests.get(f"{BLYNK_BASE_URL}/update?token={BLYNK_AUTH_TOKEN}&v1=0", timeout=5)
            requests.get(f"{BLYNK_BASE_URL}/update?token={BLYNK_AUTH_TOKEN}&v2=He_thong_on_dinh", timeout=5)
            print(">>> Reset: Hệ thống an toàn.")
        except: pass
    threading.Thread(target=task).start()

# ==========================================
# 3. KHỞI TẠO 
# ==========================================
cap = cv2.VideoCapture(SOURCE)

backSub = cv2.createBackgroundSubtractorMOG2(history=800, varThreshold=60, detectShadows=True)
kernel = np.ones((5, 5), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    warmup_frames += 1
    frame = cv2.resize(frame, (640, 480))
    display_frame = frame.copy()
    cv2.rectangle(display_frame, top_left, bottom_right, (255, 0, 0), 2)

    # Làm mờ mạnh hơn để triệt tiêu nhiễu hạt
    blurred = cv2.GaussianBlur(frame, (21, 21), 0)
    fgMask = backSub.apply(blurred)
    _, fgMask = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)
    fgMask = cv2.erode(fgMask, kernel, iterations=1)

    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    # --- NHẬN DIỆN LỬA (YCrCb) ---
    fire_m = cv2.inRange(ycrcb, np.array([160, 150, 50]), np.array([255, 255, 120]))
    fire_res = cv2.bitwise_and(fgMask, fire_m)

    has_fire = False
    msg = ""

    if warmup_frames > 80: 
        # Quét Lửa (Khung Đỏ)
        cnt_f, _ = cv2.findContours(fire_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnt_f:
            if cv2.contourArea(c) > 1000:
                x, y, w, h = cv2.boundingRect(c)
                if top_left[0] < x+w//2 < bottom_right[0] and top_left[1] < y+h//2 < bottom_right[1]:
                    has_fire = True
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 3)

    if has_fire: 
        msg = "HOA_HOAN_LUA"
        fire_counter += 1
    else:
        fire_counter = max(0, fire_counter - 1)

    now = time.time()
    if fire_counter >= CONFIRM_FRAMES and last_state == 0:
        trigger_alarm_process(display_frame, msg, False)
        last_alert_time = now; last_state = 1
    elif last_state == 1 and (now - last_alert_time > ALERT_INTERVAL) and fire_counter >= CONFIRM_FRAMES:
        trigger_alarm_process(display_frame, msg, True)
        last_alert_time = now
    elif fire_counter == 0 and last_state == 1:
        reset_system_status(); last_state = 0

    cv2.imshow("Camera Feed", display_frame)
    cv2.imshow("Fire Mask", fire_m)
    cv2.imshow("Motion Detection", fgMask)

    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()