import cv2
import numpy as np
import mediapipe as mp

from camera import Camera
import utils


def overlay_transparent(background, overlay, x, y):
    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255
            ],
            axis=2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y + h, x:x + w] = (1.0 - mask) * background[y:y + h, x:x + w] + mask * overlay_image

    return background


def main():
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

    background = cv2.imread('./img/CameraBackground2.png', cv2.IMREAD_UNCHANGED)
    like = cv2.imread('./img/like.png', cv2.IMREAD_UNCHANGED)
    dislike = cv2.imread('./img/dislike.png', cv2.IMREAD_UNCHANGED)
    peace = cv2.imread('./img/peace.png', cv2.IMREAD_UNCHANGED)
    fuck = cv2.imread('./img/fuck.png', cv2.IMREAD_UNCHANGED)
    stop = cv2.imread('./img/stop.png', cv2.IMREAD_UNCHANGED)
    dont_see = cv2.imread('./img/dontSee.png', cv2.IMREAD_UNCHANGED)

    main_frame = np.zeros_like(background)

    alpha_background = background[:, :, 3] / 255
    alpha_foreground = like[:, :, 3] / 255

    while True:
        key = cv2.waitKey(1) & 0xFF

        frame = video_getter.frame
        frame = cv2.flip(frame, 1)

        # Detection
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

        frame_c = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        frame_c_m = cv2.resize(frame_c, (509, 373))
        alpha_frame = main_frame[:, :, 3] / 255

        # Paste
        main_frame[160:160 + 720, 40:40 + 1280] = frame_c
        main_frame[507:507 + 373, 1351:1351 + 509] = frame_c_m
        #

        # Подстановка изображений на задний фон
        for color in range(0, 3):
            # Жесты
            background[:, :, color] = alpha_foreground * like[:, :, color] + \
                                      alpha_background * background[:, :, color] * (1 - alpha_foreground)
        #
            # Задний фон
            background[:, :, color] = alpha_background * background[:, :, color] + \
                                      alpha_frame * main_frame[:, :, color] * (1 - alpha_background)

        # Full screen
        background = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)
        cv2.namedWindow("Robot control", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Robot control", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Robot control', background)
        # cv2.imshow('Camera', frame_c)

        if key == ord('q'):
            video_getter.stop()
            cv2.destroyAllWindows()
            print("[#] Manual closing of the program")
            break


if __name__ == '__main__':
    main()
