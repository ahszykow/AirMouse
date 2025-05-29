import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import ctypes
import psutil
import win32gui
import win32con
import time
import subprocess
import os

# Constants
PINCH = 0.05
SMOOTH_FACTOR = 0.1
MOVE_THRESHOLD = 0.05
CLICK_COOLDOWN = 0.7  # seconds

# State
prev_x, prev_y = 0, 0
prev_wrist_x = None
osk_opened = False
last_click_time = 0

# Path to TabTip
tabtip_path = r"C:\Program Files\Common Files\Microsoft Shared\ink\TabTip.exe"
if not os.path.exists(tabtip_path):
    tabtip_path = r"C:\Windows\System32\TabTip.exe"  # fallback path

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Camera and screen
cam = cv2.VideoCapture(0)
screenWidth, screenHeight = pyautogui.size()

def euc_dist(p1, p2):
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

def make_cv2_window_topmost(window_name):
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd:
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                              win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

def is_fist(hand_landmarks):
    fingertip_ids = [8, 12, 16, 20]
    base_ids = [5, 9, 13, 17]
    folded_count = 0
    for tip_id, base_id in zip(fingertip_ids, base_ids):
        tip = hand_landmarks.landmark[tip_id]
        base = hand_landmarks.landmark[base_id]
        if euc_dist(tip, base) < 0.07:
            folded_count += 1
    return folded_count == 4

def launch_tabtip():
    try:
        # Attempt to start TabTip.exe using explorer (more reliable on some builds)
        subprocess.Popen(['explorer.exe', tabtip_path], shell=True)
        time.sleep(0.5)

        # Focus it using COM automation (if needed), or force show by toggling the taskbar
        ctypes.windll.user32.ShowWindow(ctypes.windll.user32.FindWindowW("IPTip_Main_Window", None), 5)  # SW_SHOW
    except Exception as e:
        print(f"Error launching TabTip: {e}")

def close_tabtip():
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] and 'tabtip' in proc.info['name'].lower():
            proc.kill()

while cam.isOpened():
    success, frame = cam.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    gesture_name = "None"
    fist_count = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if is_fist(hand_landmarks):
                fist_count += 1
                continue  # Skip movement and click if fist

            # Otherwise handle movement and click
            index = hand_landmarks.landmark[8]
            middle = hand_landmarks.landmark[12]
            thumb = hand_landmarks.landmark[4]
            wrist = hand_landmarks.landmark[0]

            pinch_dist = euc_dist(index, thumb)
            move_dist = euc_dist(index, middle)

            # Move cursor
            if move_dist < MOVE_THRESHOLD:
                target_x = int(index.x * screenWidth)
                target_y = int(index.y * screenHeight)
                curr_x = int(prev_x + (target_x - prev_x) * SMOOTH_FACTOR)
                curr_y = int(prev_y + (target_y - prev_y) * SMOOTH_FACTOR)
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y
                gesture_name = "Move"

            # Click (cooldown)
            now = time.time()
            if pinch_dist < PINCH and (now - last_click_time) > CLICK_COOLDOWN:
                pyautogui.click(button='left')
                last_click_time = now
                gesture_name = "Click"

    # Fist logic
    if fist_count == 1:
        gesture_name = "FIST"
    elif fist_count == 2:
        gesture_name = "DOUBLE FIST"
        if not osk_opened:
            launch_tabtip()
            osk_opened = True
        else:
            close_tabtip()
            osk_opened = False
    else:
        osk_opened = osk_opened  # maintain current state if neither 1 nor 2 fists

    # Draw gesture text
    text = f"Gesture: {gesture_name}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (0, 255, 0)
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.putText(frame, text, (w - text_width - 10, h - 10), font, font_scale, color, thickness, cv2.LINE_AA)

    # Show window
    window_name = "Gesture Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, frame)
    make_cv2_window_topmost(window_name)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
