import cv2
import mediapipe as mp
import numpy as np

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
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# intilializing webcam
cam = cv2.VideoCapture(0)

while cam.isOpened():

    # making cam read if not successful then break
    success, image = cam.read()

    if not success:
        break

    # mirror the camera and grab dims
    image = cv2.flip(image, 1)
    h, w, _ = image.shape

    # init canvas variable
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

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

            # check if user is making fist and output bool
            fist = all(fingertip.y > base.y for fingertip, base in zip(fingertips, bases))
            
            # convert index and thumb tip to pixel dims
            index_pixels = (int(index.x * w), int(index.y * h))
            thumb_pixels = (int(thumb.x * w), int(thumb.y * h))

            # calc distance between thumb and index finger
            distance = euc_dist(thumb,index)

            # check if fingers are touching using threshold
            if distance < PINCH:
                drawing_enabled = True  # start drawing
            elif fist and canvas[:].any() != 0:
                # STILL TESTING: WILL CLEAR CANVAS AND EXPORT/REMOVE CANVAS TO IMAGE
                white_bg = np.ones_like(canvas, dtype=np.uint8) * 255
                drawing_on_white = cv2.addWeighted(white_bg, 1, canvas, 1, 0)

                # Save the image
                filename = f"drawing_{save_counter}.png"
                cv2.imwrite(filename, drawing_on_white)
                print(f"Drawing saved as {filename}")

                # Open saved image in new window
                saved_image = cv2.imread(filename)
                cv2.imshow(f"Saved Drawing {save_counter}", saved_image)
                save_counter += 1
                
                canvas[:] = 0  # clear canvas if fist is detected
                drawing_enabled = False  # reset drawing flag
                prev_x, prev_y = None, None  # reset previous drawing point
            else:
                drawing_enabled = False  # stop drawing
                prev_x, prev_y = None,None  # reset drawing position

            # if drawing is enabled let user draw on canvas and update prev variables for canvas purposes
            if drawing_enabled:
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (index_pixels[0], index_pixels[1]), (0, 255, 0), 10)
                prev_x, prev_y = index_pixels[0], index_pixels[1]

    # merge image and drawing canvas
    combined = cv2.addWeighted(image, 1, canvas, 1, 0)

    # display the image in window and set window z-index to top
    cv2.imshow('Drawing with Index Finger', combined)
    cv2.setWindowProperty('Drawing with Index Finger', cv2.WND_PROP_TOPMOST, 1)

    # let user tap esc to exit cam or 'c' to clear drawing
    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # ESC key
        break
    elif key == ord('c'):  
        canvas[:] = 0

cam.release()
cv2.destroyAllWindows()
