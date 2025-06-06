import cv2
import mediapipe as mp
import pyautogui
import webbrowser
import numpy as np

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands(max_num_hands=1)
face_mesh = mp_face_mesh.FaceMesh()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_point(landmarks, idx, w, h):
    return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            y = hand_landmarks.landmark[8].y
            if y < 0.4:
                pyautogui.press('volumeup')
            elif y > 0.6:
                pyautogui.press('volumedown')

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
            
            landmarks = face_landmarks.landmark

            # Nose tip for face left/right
            nose_tip = landmarks[1]
            if nose_tip.x < 0.4:
                pyautogui.hotkey('ctrl', 'left')
            elif nose_tip.x > 0.6:
                pyautogui.hotkey('ctrl', 'right')

            # Blink detection (eyes: 159-145 left, 386-374 right)
            left_eye = [get_point(landmarks, i, w, h) for i in [159, 145]]
            right_eye = [get_point(landmarks, i, w, h) for i in [386, 374]]
            left_dist = euclidean(*left_eye)
            right_dist = euclidean(*right_eye)
            if left_dist < 5 and right_dist > 5:
                pyautogui.click()

            # Eyebrow raise (compare eyebrow to eye)
            left_brow = get_point(landmarks, 105, w, h)
            left_eye_top = get_point(landmarks, 159, w, h)
            if abs(left_brow[1] - left_eye_top[1]) > 20:
                pyautogui.scroll(-20)

            # Smile detection (mouth corners: 61, 291)
            mouth_left = get_point(landmarks, 61, w, h)
            mouth_right = get_point(landmarks, 291, w, h)
            mouth_width = euclidean(mouth_left, mouth_right)
            if mouth_width > 120:
                webbrowser.open("https://www.google.com")

            # Mouth open (top 13, bottom 14)
            top_lip = get_point(landmarks, 13, w, h)
            bottom_lip = get_point(landmarks, 14, w, h)
            if euclidean(top_lip, bottom_lip) > 30:
                pyautogui.hotkey('win', 'up')  # Maximize window

    cv2.imshow("Face + Hand Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()