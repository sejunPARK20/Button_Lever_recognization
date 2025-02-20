# 250107-250110 Butten_Recognition (Azure Kinect, MediaPipe, OpenCV)

import cv2
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A
import mediapipe as mp
import time

# 1. Setting Hand recognition & Kinect
 # 1-1. Mediapipe 손 인식 설정
mp_hands = mp.solutions.hands           # 손 찾기 기능
mp_drawing = mp.solutions.drawing_utils # 손 그려주는 기능

 # 1-2. Azure Kinect 초기화
k4a = PyK4A(
    Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        synchronized_images_only=True,
    )
)
k4a.start()

# 2. MediaPipe Hands 객체 설정
hands = mp_hands.Hands(
    max_num_hands = 2,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)

start_time = None
fixed_buttons = []

lower_yellowButton = np.array([20, 50, 180])
upper_yellowButton = np.array([80, 255, 255])

start_time = time.time()
calibration_done = False

# Button 좌표 저장용
button_positions = {}

# 3. Button 인식 & Contour 그린 후 좌표 저장
while True:
    capture = k4a.get_capture()
    frame = capture.color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    if not calibration_done:
        # Yellow영역 추출
        mask = cv2.inRange(hsv, lower_yellowButton, upper_yellowButton)
        
        # Contour 추출
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 버튼 위치 저장
        detected_positions = []
        for contour in contours:
            if cv2.contourArea(contour) < 100:
                continue
            b_x, b_y, b_w, b_h = cv2.boundingRect(contour)
            detected_positions.append((b_x, b_y, b_w, b_h))
        
        # 3초 후 버튼 위치 고정
        if time.time() - start_time >= 3:
            detected_positions.sort(key=lambda pos: (pos[1], pos[0]))  # Y, X 기준 정렬
            button_positions = {i + 1: pos for i, pos in enumerate(detected_positions[:26])}
            calibration_done = True
            print("★★★ Button positions fixed ★★★")
            for num, pos in button_positions.items():
                print(f"Button {num}: {pos}")

    if calibration_done:
         # 저장된 버튼 좌표에 따라 사각형 그리기
        for button_number, (b_x, b_y, b_w, b_h) in button_positions.items():
            cv2.rectangle(frame, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0), 2)
            cv2.putText(frame, str(button_number), (b_x, b_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    if time.time() - start_time >= 5:
        break

# 4. 검지 끝이 어느 버튼 위에 있는지 판단하는 함수
def button_recognition(cx, cy, button_positions, frame):
    # 검지 끝 위치(cx, cy)가 버튼 영역에 있는지 확인
    for button_number, (b_x, b_y, b_w, b_h) in button_positions.items():
        if b_x <= cx <= b_x + b_w and b_y <= cy <= b_y + b_h:
            # print(f"Button {button_number} Pressed!")
            cv2.putText(frame, f"Button {button_number}",(85, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
            return

# 5. ★★★ Camera 작동 및 MediaPipe로 손동작 인식 ★★★
h, w, _ = frame.shape # (720, 1280, _ )
try:
    while True:
        # 5-1. Kinect에서 RGB, Depth image 가져오기
        capture = k4a.get_capture()
        rgb_img = capture.color
        depth_img = capture.transformed_depth
        
        if capture.color is None or capture.depth is None:
            continue # 인식을 못하면 루프 벗어나기
        
        # 5-2. MediaPipe로 손동작 인식
        
         # 입력 받은 이미지를 RGB로 전환 (Mediapipe는 RGB값을 사용함)
        frame = cv2.cvtColor(capture.color, cv2.COLOR_BGR2RGB)
        
         # MediaPipe로 손동작 인식
        results = hands.process(frame)

         # RGB를 다시 BGR로 전환 (OpenCV에서 다시 시각화)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)             
                             
        # 5-3. Hand에 Landmark 그리기
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                
                # Lanmark 8번(검지 끝) 좌표 저장
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # 3검지 끝 Text 좌표 설정 및 출력
                
                 # 검지 끝 x,y 좌표
                cx, cy = int(index_finger_tip.x*w), int(index_finger_tip.y*h)
                
                 # 검지 끝 깊이 좌표
                cz = depth_img[cy, cx]
                
                 # 검지 끝 -> 표시
                cv2.circle(frame,(cx, cy), 7, (0, 255, 0), -1)     
                
                # 버튼 위치 추정
                button_recognition(cx, cy, button_positions, frame)

                # 좌표 출력
                print(f"RGB: ({cx}, {cy},{index_finger_tip.z}) Depth: ({cz})")

         # 5-4. 저장된 버튼 좌표에 따라 사각형 그리기
        for button_number, (b_x, b_y, b_w, b_h) in button_positions.items():
            cv2.rectangle(frame, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0), 2)
            cv2.putText(frame, str(button_number), (b_x, b_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 5-5. 결과 출력
        cv2.imshow('Azure kinect + MediaPipe', frame)
        cv2.imshow("HSV Mask", mask)
        
        # 'q' 눌러서 루프 나오기
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
        
finally:
    # Kinect, MediaPipe, OpenCV 종료
    k4a.stop()
    hands.close()
    cv2.destroyAllWindows()