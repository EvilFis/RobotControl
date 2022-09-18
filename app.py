
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

pTime = 0 # Для расчёта fps
show_fps = False # Для показа fps
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


while True:

    finger_count = {"RIGHT": 0, "LEFT": 0} # Счетчик пальцев

    # Статус пальцев
    finger_status = {'RIGHT_THUMB': False, "RIGHT_INDEX": False,
                     'RIGHT_RING': False, 'RIGHT_MIDDLE': False,
                     'RIGHT_PINKY': False, 'LEFT_THUMB': False,
                     'LEFT_INDEX': False, 'LEFT_RING': False,
                     'LEFT_MIDDLE': False, 'LEFT_PINKY': False}

    key = cv2.waitKey(1) & 0xFF

    # Получаем видеопоток и отражаем по горизонтале изображение
    frame = video_getter.frame
    frame = cv2.flip(frame, 1)

    # переводим изображение в RGB и запускаем поиск точек руки
    dFrame = cv2.cvtColor(frame[:325, :490], cv2.COLOR_BGR2RGB)
    results = hand.process(dFrame)

    cv2.line(frame, (490, 0), (490, 325), (255, 255, 0), 3)
    cv2.line(frame, (490, 325), (0, 325), (255, 255, 0), 3)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks # Список точек
        hand_label = results.multi_handedness[0].classification[0].label # Название руки
        frame_k = get_frame_keypoints(hand_landmarks, frame[:325, :490]) # Определение координат точек пальцев на изображении
        # Получаем все координаты по Х и У
        all_x = [i[0] for i in frame_k]
        all_y = [i[1] for i in frame_k]

        # Рисуем коробку с выводом названия руки
        cv2.rectangle(frame, (min(all_x) - 15, min(all_y) - 15), (max(all_x) + 15, max(all_y) + 15), (255, 0, 0), 3)
        cv2.putText(frame, hand_label, (min(all_x) - 30, min(all_y) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)

        # Делаем коннектор к руке
        for connect in mp_hands.HAND_CONNECTIONS:
            cv2.line(frame, frame_k[connect[0]], frame_k[connect[1]], (255, 255, 255), 2)

        for center in frame_k:
            cv2.circle(frame, center, 3, (0, 0, 255), cv2.FILLED)

        for tip_index in finger_tips:
            finger_name = tip_index.name.split('_')[0]
            if frame_k[tip_index][1] < frame_k[tip_index - 2][1]:
                finger_status[hand_label.upper()+'_'+finger_name] = True
                finger_count[hand_label.upper()] += 1
            else:
                finger_status[hand_label.upper()+'_'+finger_name] = False

        # Определяем поднятый большой палец
        thumb_tip_x = frame_k[mp_hands.HandLandmark.THUMB_TIP][0]
        thumb_tip_y = frame_k[mp_hands.HandLandmark.THUMB_TIP][1]

        thumb_mcp_x = frame_k[mp_hands.HandLandmark.THUMB_TIP - 2][0]
        thumb_mcp_y = frame_k[mp_hands.HandLandmark.THUMB_TIP - 2][1]

        if max(all_x) - min(all_x) < max(all_y) - min(all_y):
            if hand_label == "Right" and thumb_tip_x < thumb_mcp_x or hand_label == "Left" and thumb_tip_x > thumb_mcp_x:
                finger_status[hand_label.upper()+'_THUMB'] = True
                finger_count[hand_label.upper()] += 1
        elif max(all_x) - min(all_x) > max(all_y) - min(all_y):
            if thumb_tip_y < thumb_mcp_y:
                finger_status[hand_label.upper()+'_THUMB'] = True
                finger_count[hand_label.upper()] += 1

    if key == ord('d'):
        show_fps = not show_fps

    if show_fps:
        cTime = time.time()
        fps = str(int(1 / (cTime - pTime)))
        pTime = cTime

        cv2.putText(frame, fps, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

    cv2.imshow('Stream', frame)

    if key == ord('q'):
        video_getter.stop()
        cv2.destroyAllWindows()
        print("[#] Manual closing of the program")
        break
