import cv2
import mediapipe as mp
import numpy as np
import time

from camera import Camera
import utils


def main():
    # Настройка распознование рук
    mp_hands = mp.solutions.hands
    hand = mp_hands.Hands(max_num_hands=1,
                          min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)

    # Получаем видеопоток
    video_getter = Camera(0, 1280, 720).start()

    # Выписываем индексы необходимых пальцев
    finger_tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                   mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    wrist = mp_hands.HandLandmark.WRIST
    finger_middle_mcp = mp_hands.HandLandmark.MIDDLE_FINGER_MCP

    # Загрузка изображений для дальнейшей подстановки
    like = cv2.imread('./img/like.png')
    dislike = cv2.imread('./img/dislike.png')
    peace = cv2.imread('./img/peace.png')
    fuck = cv2.imread('./img/fuck.png')
    stop = cv2.imread('./img/stop.png')

    # Делаем скрин первого кадра
    bad_person = cv2.resize(video_getter.frame, (499, 363))

    save_code = ''  # Вспомогательная переменная

    while True:

        start_time = time.time()

        background = cv2.imread('./img/dontSee.png')
        code = '-1\n'

        # Настройка жестов
        finger_count = 0
        finger_status = {'THUMB': False,
                         "INDEX": False,
                         'RING': False,
                         'MIDDLE': False,
                         'PINKY': False}

        # Прослушка клавиш
        key = cv2.waitKey(1) & 0xFF

        # read image
        frame = cv2.cvtColor(cv2.flip(video_getter.frame, 1), cv2.COLOR_BGR2RGB)

        detection_region = frame[0:450, 720:1280]

        results = hand.process(detection_region)
        hand_landmarks = results.multi_hand_landmarks

        if hand_landmarks:
            hand_label = results.multi_handedness[0].classification[0].label
            frame_k = utils.get_frame_keypoints(hand_landmarks, detection_region)

            for connect in mp_hands.HAND_CONNECTIONS:
                cv2.line(detection_region, frame_k[connect[0]], frame_k[connect[1]], (255, 255, 255), 2)

            for center in frame_k:
                cv2.circle(detection_region, center, 3, (0, 0, 255), cv2.FILLED)

            rotation_hand = utils.orientation(frame_k[wrist], frame_k[finger_middle_mcp])

            # Vertical hand
            if rotation_hand == 'U' or rotation_hand == 'D':
                for tip_index in finger_tips:
                    if frame_k[tip_index][1] < frame_k[tip_index - 2][1]:
                        finger_status[tip_index.name.split('_')[0]] = True
                        finger_count += 1

            # horizontal
            elif rotation_hand == 'R' or rotation_hand == 'L':
                for tip_index in finger_tips:
                    if frame_k[tip_index][0] < frame_k[tip_index - 2][0] and rotation_hand == 'L':
                        finger_status[tip_index.name.split('_')[0]] = True
                        finger_count += 1
                    elif frame_k[tip_index][0] > frame_k[tip_index - 2][0] and rotation_hand == 'R':
                        finger_status[tip_index.name.split('_')[0]] = True
                        finger_count += 1

            status, side = utils.find_thumb(frame_k, hand_label, rotation_hand)
            finger_status['THUMB'] = status
            finger_count += status

            # Find gesture
            # Peace
            if finger_count == 2 and finger_status['MIDDLE'] and finger_status['INDEX'] and rotation_hand == 'U':
                background = peace
                code = '3\n'

            # Bad finger
            elif 1 <= finger_count <= 2 and finger_status['MIDDLE'] and rotation_hand == 'U':
                background = fuck
                bad_person = frame

                # Все координаты по оси Х и У
                all_x = [i[0] for i in frame_k]
                all_y = [i[1] for i in frame_k]
                cv2.rectangle(detection_region, (min(all_x) - 30, min(all_y) - 20), (max(all_x) + 30, max(all_y) + 20),
                              (0, 0, 0), cv2.FILLED)

                bad_person = cv2.cvtColor(bad_person, cv2.COLOR_RGB2BGR)
                bad_person = cv2.resize(bad_person, (499, 363))
                code = '4\n'

            # like
            elif finger_count == 1 and finger_status['THUMB'] and rotation_hand == 'R' and hand_label == 'Right' \
                    and side or finger_count == 1 and finger_status['THUMB'] and rotation_hand == 'L' \
                    and hand_label == 'Right' and side:
                background = like
                code = '1\n'

            # dislike
            elif finger_count == 1 and finger_status['THUMB'] and rotation_hand == 'R' and hand_label == 'Right' \
                    and not side or finger_count == 1 and finger_status['THUMB'] and rotation_hand == 'L' \
                    and hand_label == 'Right' and not side:
                background = dislike
                code = '2\n'

            # like
            elif finger_count == 1 and finger_status['THUMB'] and rotation_hand == 'R' and hand_label == 'Left' \
                    and not side or finger_count == 1 and finger_status['THUMB'] and rotation_hand == 'L' \
                    and hand_label == 'Left' and not side:
                background = like
                code = '1\n'

            # dislike
            elif finger_count == 1 and finger_status['THUMB'] and rotation_hand == 'R' and hand_label == 'Left' \
                    and side or finger_count == 1 and finger_status['THUMB'] and rotation_hand == 'L' \
                    and hand_label == 'Left' and side:
                background = dislike
                code = '2\n'

            # Stop
            elif finger_count == 5 and rotation_hand == 'U' and side:
                background = stop
                code = '0\n'

            # Отправка данных на сервер
            if code != save_code and code != '-1\n':
                cv2.putText(frame, f'Pernul {code}', (400, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                save_code = code

        # Если рука не распознана, то отправляем -1
        elif code != save_code and code == '-1\n':
            cv2.putText(frame, f'Pernul ne to', (400, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            save_code = code

        # FPS
        cv2.putText(frame, f'{int(1 / (time.time() - start_time))}', (50, 100), cv2.FONT_HERSHEY_COMPLEX, 2,
                    (255, 0, 0), 1)

        # Paste frame
        frame = cv2.resize(frame, (1270, 710))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        background[166:166 + 710, 45:45 + 1270] = frame
        background[513:513 + 363, 1358:1358 + 499] = bad_person

        cv2.line(background, (760, 166), (760, 615), color=(255, 0, 0), thickness=2)
        cv2.line(background, (760, 615), (1314, 615), color=(255, 0, 0), thickness=2)

        # Full screen
        cv2.namedWindow("Robot control", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Robot control", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Robot control', background)

        # Обработка клавиш
        if key == ord('q'):
            video_getter.stop()
            cv2.destroyAllWindows()
            print("[#] Manual closing of the program")
            break


if __name__ == '__main__':
    main()
