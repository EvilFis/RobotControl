import cv2
import mediapipe as mp
import numpy as np

from camera import Camera
import utils


def main():
    mp_hands = mp.solutions.hands
    hand = mp_hands.Hands(max_num_hands=1,
                          min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)

    video_getter = Camera(0, 1280, 720).start()

    finger_tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                   mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]

    finger_thumb = mp_hands.HandLandmark.THUMB_TIP
    wrist = mp_hands.HandLandmark.WRIST
    finger_middle_mcp = mp_hands.HandLandmark.MIDDLE_FINGER_MCP

    like = cv2.imread('./img/like.png', cv2.IMREAD_UNCHANGED)
    dislike = cv2.imread('./img/dislike.png', cv2.IMREAD_UNCHANGED)
    peace = cv2.imread('./img/peace.png', cv2.IMREAD_UNCHANGED)
    fuck = cv2.imread('./img/fuck.png', cv2.IMREAD_UNCHANGED)
    stop = cv2.imread('./img/stop.png', cv2.IMREAD_UNCHANGED)
    dont_see = cv2.imread('./img/dontSee.png', cv2.IMREAD_UNCHANGED)

    test = cv2.imread('./img/test.png')

    bad_person = cv2.resize(cv2.cvtColor(video_getter.frame, cv2.COLOR_RGB2BGRA), (499, 363))

    while True:
        frame_cap_flag = False
        background = cv2.imread('./img/CameraBackground2.png', cv2.IMREAD_UNCHANGED)
        png = dont_see

        # Настройка жестов
        gesture = ''
        finger_count = 0
        finger_status = {'THUMB': False,
                         "INDEX": False,
                         'RING': False,
                         'MIDDLE': False,
                         'PINKY': False}

        # Прослушка клавиш
        key = cv2.waitKey(1) & 0xFF

        # Перевод заднего изображения в BGRA
        background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)

        # read image
        frame = cv2.cvtColor(cv2.flip(video_getter.frame, 1), cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (1270, 710))

        results = hand.process(frame)
        hand_landmarks = results.multi_hand_landmarks

        if hand_landmarks:
            hand_label = results.multi_handedness[0].classification[0].label
            frame_k = utils.get_frame_keypoints(hand_landmarks, frame)

            for connect in mp_hands.HAND_CONNECTIONS:
                cv2.line(frame, frame_k[connect[0]], frame_k[connect[1]], (255, 255, 255), 2)

            for center in frame_k:
                cv2.circle(frame, center, 3, (0, 0, 255), cv2.FILLED)

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
                gesture = 'Peace'
                png = peace

            # Bad finger
            elif 1 <= finger_count <= 2 and finger_status['MIDDLE'] and rotation_hand == 'U':
                gesture = ':('
                png = fuck

                bad_person = frame
                frame_cap_flag = True

                # Все координаты по оси Х и У
                all_x = [i[0] for i in frame_k]
                all_y = [i[1] for i in frame_k]
                cv2.rectangle(bad_person, (min(all_x) - 30, min(all_y) - 20), (max(all_x) + 30, max(all_y) + 20),
                              (0, 0, 0), cv2.FILLED)

                bad_person = cv2.resize(bad_person, (499, 363))

            # like
            elif finger_count == 1 and finger_status['THUMB'] and rotation_hand == 'R' and hand_label == 'Right' \
                    and side or finger_count == 1 and finger_status['THUMB'] and rotation_hand == 'L' \
                    and hand_label == 'Right' and side:

                gesture = 'It\'s like :D'
                png = like

            # dislike
            elif finger_count == 1 and finger_status['THUMB'] and rotation_hand == 'R' and hand_label == 'Right' \
                    and not side or finger_count == 1 and finger_status['THUMB'] and rotation_hand == 'L' \
                    and hand_label == 'Right' and not side:

                gesture = 'It\'s dislike :<'
                png = dislike

            # like
            elif finger_count == 1 and finger_status['THUMB'] and rotation_hand == 'R' and hand_label == 'Left' \
                    and not side or finger_count == 1 and finger_status['THUMB'] and rotation_hand == 'L' \
                    and hand_label == 'Left' and not side:

                gesture = 'It\'s like :D'
                png = like

            # dislike
            elif finger_count == 1 and finger_status['THUMB'] and rotation_hand == 'R' and hand_label == 'Left' \
                    and side or finger_count == 1 and finger_status['THUMB'] and rotation_hand == 'L' \
                    and hand_label == 'Left' and side:

                gesture = 'It\'s dislike :<'
                png = dislike

            # Stop
            elif finger_count == 5 and rotation_hand == 'U' and side:
                gesture = 'Stop'
                png = stop

            cv2.putText(frame, f'Label: {hand_label}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f'Gesture: {gesture}', (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Finger_counts: {finger_count}", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
                        2)
            cv2.putText(frame, f'Rotate: {rotation_hand}', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Подстановка изображений на задний фон
        alpha_background = utils.get_alpha(background)
        alpha_foreground = utils.get_alpha(png)
        background = utils.get_background_img(background, png, alpha_background, alpha_foreground)

        # Paste frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)
        background[166:166 + 710, 45:45 + 1270] = frame

        if frame_cap_flag:
            bad_person = cv2.cvtColor(bad_person, cv2.COLOR_RGB2BGRA)

        background[513:513+363, 1358:1358+499] = bad_person

        # Full screen
        background = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)
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
