import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# temp variables for testing + canvas variables
save_counter = 1
canvas = None
prev_x, prev_y = None, None

# pinching threshold to change as needed
PINCH = 0.05

# helper function to calculate distance of fingers
def euc_dist(fing1,fing2):
    return np.linalg.norm(np.array([fing1.x, fing1.y]) - np.array([fing2.x, fing2.y]))

# initializing mediapide hands library
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# intilializing webcam
cam = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# Screen sizes
screenWidth, screenHeight = pyautogui.size()

# Prevent multiple clicks
clicked = False

while cam.isOpened():

    # making cam read if not successful then break
    success, image = cam.read()

    if not success:
        break

    # mirror the camera and grab dims
    image = cv2.flip(image, 1)
    h, w, _ = image.shape

    # convert cam image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # process the image and detect hands
    results = hands.process(image_rgb)

    # if we detect hands then run main program
    if results.multi_hand_landmarks:

        # iterate through hand landmarks 
        for hand_landmarks in results.multi_hand_landmarks:

            ## COMMENT OUT: IF YOU DONT WANT FULL HAND LANDMARKS DISPLAYED
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # identifying fingertips and bases based on https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
            fingertips = [hand_landmarks.landmark[i] for i in [8, 12, 16, 20]]  # index, middle, ring, pinky
            bases = [hand_landmarks.landmark[i] for i in [5, 9, 13, 17]]  # corresponding joints

            # fetching index and thumb tip landmark points
            index = hand_landmarks.landmark[8]
            thumb = hand_landmarks.landmark[4]

            # calc distance between thumb and index finger
            distance = euc_dist(thumb,index)
            
            x = int(index.x * screenWidth)
            y = int(index.y * screenHeight)

            pyautogui.moveTo(x,y)

            if distance < PINCH and not clicked:
                pyautogui.click(button='left')
                clicked = True
            elif distance >= PINCH:
                clicked = False




    # let user tap esc to exit cam or 'c' to clear drawing
    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # ESC key
        break


cam.release()
cv2.destroyAllWindows()
