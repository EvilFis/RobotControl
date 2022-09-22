import time

import cv2
import mediapipe as mp
import numpy as np

from camera import Camera

# Определение библиотеки по определению руки
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(max_num_hands=1,
                      model_complexity=0,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7)

# Самописная библиотека по захвату видео потока
video_getter = Camera(0, 1280, 720).start()

pTime = 0  # Для расчёта fps
show_fps = False  # Для показа fps
index = 0

# Список точек необходимых пальцев
finger_tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
               mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]


# Функция по определению координат точек пальцев на изображении
def get_frame_keypoints(landmarks, frame):
    frame_keypoints = []
    # print(landmarks)
    for ldm in landmarks:
        for p in range(21):
            pxl_x = int(round(frame.shape[1] * ldm.landmark[p].x))
            pxl_y = int(round(frame.shape[0] * ldm.landmark[p].y))
            kpts = (pxl_x, pxl_y)
            frame_keypoints.append(kpts)

    return frame_keypoints


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hand = mp_hands.Hands(max_num_hands=1,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7)

while True:

    start_time = time.time()

    key = cv2.waitKey(1) & 0xFF

    # read image
    frame = cv2.cvtColor(cv2.flip(video_getter.frame, 1), cv2.COLOR_BGR2RGB)

    frame.flags.writeable = False
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.

    cv2.putText(image, f'FPS: {round(1 / (time.time() - start_time), 2)}', (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 0, 0), 2)
    # Full screen
    cv2.namedWindow("Robot control", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Robot control", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Robot control', image)

    if key == ord('q'):
        video_getter.stop()
        cv2.destroyAllWindows()
        print("[#] Manual closing of the program")
        break
