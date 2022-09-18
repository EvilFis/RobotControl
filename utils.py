def get_frame_keypoints(landmarks, frame):
    frame_keypoints = []
    for ldm in landmarks:
        for p in range(21):
            pxl_x = int(round(frame.shape[1] * ldm.landmark[p].x))
            pxl_y = int(round(frame.shape[0] * ldm.landmark[p].y))
            kpts = (pxl_x, pxl_y)
            frame_keypoints.append(kpts)

    return frame_keypoints


# Определение поворота
def orientation(cord_wrist, cord_middle_mcp):
    x_wrist, y_wrist = cord_wrist[0], cord_wrist[1]
    x_middle_mcp, y_middle_mcp = cord_middle_mcp[0], cord_middle_mcp[1]

    if abs(x_middle_mcp - x_wrist) < 0.05:
        m = 1000000000
    else:
        m = abs((y_middle_mcp - y_wrist) / (x_middle_mcp - x_wrist))

    if 0 <= m <= 1:
        if x_middle_mcp > x_wrist:
            return 'R'
        else:
            return 'L'

    if m > 1:
        if y_middle_mcp < y_wrist:
            return 'U'
        else:
            return 'D'


# Поиск большого пальца
def find_thumb(keypoints, label, rotation, thumb=4, wrist_p=0):
    status = False
    index = 0
    side = True

    if rotation == 'R' or rotation == 'L':
        index = 1

    # Right
    if keypoints[thumb][index] < keypoints[wrist_p][index] and label == 'Right':
        if keypoints[thumb][index] < keypoints[thumb - 2][index]:
            status = True
            side = True

    elif keypoints[thumb][index] > keypoints[wrist_p][index] and label == 'Right':
        if keypoints[thumb][index] > keypoints[thumb - 2][index]:
            status = True
            side = False

    # Left
    elif keypoints[thumb][index] > keypoints[wrist_p][index] and label == 'Left':
        if keypoints[thumb][index] > keypoints[thumb - 2][index]:
            status = True
            side = True

    elif keypoints[thumb][index] < keypoints[wrist_p][index] and label == 'Left':
        if keypoints[thumb][index] < keypoints[thumb - 2][index]:
            status = True
            side = False

    return status, side