import cv2
import mediapipe as mp

from camera import Camera
import utils


# Определение библиотеки по определению руки
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(max_num_hands=1,
                      model_complexity=0,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7)

video_getter = Camera(0, 1280, 720).start()

finger_tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
               mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]

finger_thumb = mp_hands.HandLandmark.THUMB_TIP
wrist = mp_hands.HandLandmark.WRIST
finger_middle_mcp = mp_hands.HandLandmark.MIDDLE_FINGER_MCP


while True:

    gesture = ''
    finger_count = 0
    finger_status = {'THUMB': False,
                     "INDEX": False,
                     'RING': False,
                     'MIDDLE': False,
                     'PINKY': False}

    key = cv2.waitKey(1) & 0xFF

    frame = video_getter.frame
    frame = cv2.flip(frame, 1)

    # переводим изображение в RGB и запускаем поиск точек руки
    dFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand.process(dFrame)
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

        # Peace
        if finger_count == 2 and finger_status['MIDDLE'] and finger_status['INDEX'] and rotation_hand == 'U':
            gesture = 'Peace'

        # Bad finger
        elif 1 <= finger_count <= 2 and finger_status['MIDDLE'] and rotation_hand == 'U':
            gesture = ':('

        # like
        elif finger_count == 1 and finger_status['THUMB'] and rotation_hand == 'R' and hand_label == 'Right' and side \
                or finger_count == 1 and finger_status['THUMB'] and rotation_hand == 'L' and hand_label == 'Right' and side:
            gesture = 'It\'s like :D'

        # dislike
        elif finger_count == 1 and finger_status['THUMB'] and rotation_hand == 'R' and hand_label == 'Right' and not side \
                or finger_count == 1 and finger_status['THUMB'] and rotation_hand == 'L' and hand_label == 'Right' and not side:
            gesture = 'It\'s dislike :<'

        # like
        elif finger_count == 1 and finger_status['THUMB'] and rotation_hand == 'R' and hand_label == 'Left' and not side \
                or finger_count == 1 and finger_status['THUMB'] and rotation_hand == 'L' and hand_label == 'Left' and not side:
            gesture = 'It\'s like :D'

        # dislike
        elif finger_count == 1 and finger_status['THUMB'] and rotation_hand == 'R' and hand_label == 'Left' and side \
                or finger_count == 1 and finger_status['THUMB'] and rotation_hand == 'L' and hand_label == 'Left' and side:
            gesture = 'It\'s dislike :<'

        # Stop
        elif finger_count == 5 and rotation_hand == 'U' and side:
            gesture = 'Stop'


        cv2.putText(frame, f"Finger_counts: {finger_count}", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f'Label: {hand_label}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f'Rotate: {rotation_hand}', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f'Gesture: {gesture}', (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    if key == ord('q'):
        video_getter.stop()
        cv2.destroyAllWindows()
        print("[#] Manual closing of the program")
        break

    cv2.imshow('Main stream', frame)
