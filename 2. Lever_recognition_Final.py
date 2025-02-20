# 250107-250110 Butten_Recognition (Azure Kinect, MediaPipe, OpenCV)

import cv2
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A
import mediapipe as mp
import time

# 1. Kinect & Initial setting
 # 1-2. Azure Kinect 초기화
k4a = PyK4A(
    Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        synchronized_images_only=True,
    )
)
k4a.start()

 # 1-3. HSV & Depth 범위 설정
min_depth = 10
max_depth = 440

lower_purple = (110, 70, 120)
upper_purple = (160, 255, 255) 

kernal = np.ones((5,5), np.uint8)

start_time = time.time()
calibration_done = False

center_height = None
top_height = None
bot_height = None

# 2. average y 값 계산 함수
def average_y(contours):
    y_val = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cy = int(M["m01"] / M['m00'])
            y_val.append(cy)
    return np.mean(y_val) if y_val else None

# 3. Lever 단계 예측 함수
def predict_level(center_height, top_height, bot_height, lever_y):
    levels = {}
     
    # 3-1. center, top 5등분
    top_step = (center_height - top_height) / 5
    levels[0] = top_height + top_step/2
    
    for i in range(1,5):
        levels[i] = top_height + (i+0.5) * top_step
    
    # 3-2. center, bot 5등분
    bot_step = (bot_height - center_height) / 5
    for i in range(1,5):
        levels[i+4] = center_height + (i-0.5) * bot_step
    
    levels[9] = bot_height - bot_step/2
    
    # 3-3. Lever 단계 판단
    if lever_y <= levels[0]:
        return "10"
    elif lever_y >= levels[9]:
        return "-10"
    else:
        for i in range(0,9): 
            if i in levels and i + 1 in levels:         # 키 존재 여부 확인
                if levels[i] <= lever_y < levels[i+1]:
                    return f"{8 - i*2}"                 # '현재 Lever 단계' return
        return "Cannot Predict"                         # 전부 해당하지 않는 경우
    
# 4. Lever 인식 & Lever 단계 예측 시작
try:
    while True:
        # 4-1. Kinect에서 RGB, Depth image 가져오기
        capture = k4a.get_capture()
        rgb_img = capture.color
        frame = capture.color
        depth_img = capture.transformed_depth

         # 인식을 못하면 루프 벗어나기
        if capture.color is None or capture.depth is None:
            continue         
        
         # Depth_mask
        depth_array = np.array(depth_img, dtype = np.uint16)
        depth_mask = cv2.inRange(depth_array, min_depth, max_depth)
                
         # Purple_mask
        hsv_frame = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
        purple_mask = cv2.inRange(hsv_frame, lower_purple, upper_purple)
        
         # Combined_mask (Purple + Depth)
        combined_mask = cv2.bitwise_and(purple_mask, depth_mask)

        # 4-2. Contour 검출
         # Combined Mask 전처리
        processed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernal)        # 작은 노이즈 제거
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernal)      # 구멍 채우기
        processed_mask = cv2.GaussianBlur(processed_mask, (5, 5), 0)                    # 블러로 경계 부드럽게
        
         # 전처리된 'processed_mask'를 이용해 Contour 검출
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 4-3. Center_height, Top_height, Bot_height 측정
         # 현재 시간(elapsed_time) 계산
        current_time = time.time()
        elapsed_time = current_time - start_time
        
         # 평균 Center_height, Top_height, Bot_height 계산 & 저장
        cv2.putText(rgb_img, f"Elapsed time: {int(elapsed_time)}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

         # Center_hight 인식 & 계산 (5초  ~ 35초)
        if 5 < elapsed_time < 10:
            cv2.putText(rgb_img, f"Set the Lever at 'Center'", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            
        elif 10<= elapsed_time < 15:
            if center_height is None:
                center_height = average_y(contours)
            cv2.putText(rgb_img, f"Calculating", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
         
         # Top_hight 인식 & 계산       
        elif 15<= elapsed_time < 20:
            cv2.putText(rgb_img, f"Set the Lever at 'Top'", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)    
                       
        elif 20<= elapsed_time < 25:
            if top_height is None:
                top_height = average_y(contours)
            cv2.putText(rgb_img, f"Calculating", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
         # Bot_hight 인식 & 계산
        elif 25<= elapsed_time < 30:
            cv2.putText(rgb_img, f"Set the Lever at 'Bot'", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)  
    
        elif 30<= elapsed_time < 35:
            if bot_height is None:
                bot_height = average_y(contours)
            cv2.putText(rgb_img, f"Calculating", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
         # 3가지 height 인식 후 각각의 결과값(평균 y값) 출력
        elif 35 <= elapsed_time:
            cv2.putText(rgb_img, f"1. Top_height = {top_height}", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(rgb_img, f"2. Center_height = {center_height}", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(rgb_img, f"3. Bot_height = {bot_height}", (50,200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            if lever_contour is not None:
                M = cv2.moments(lever_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    if center_height and top_height and bot_height:
                        level = predict_level(center_height, top_height, bot_height, cy)
                        cv2.putText(rgb_img, f"Level = {level}", (850,100), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 5)
    
        # 4-4. Lever Contour의 중심점 계산 & 시각화
        lever_contour = None
        
         # Contour x,y,w,h 값 저장
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            predicted_ratio = w/h
            area = cv2.contourArea(contour)
            
            # 특정 조건에 맞는 윤곽선만 선택
            if 0.8 < predicted_ratio < 4.0  and 1000 < area < 10000:
                lever_contour = contour
                break

         # 중심좌표 시각화
        if lever_contour is not None:
            M = cv2.moments(lever_contour)

            # M["m00"]가 0인지 확인
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                print(f"Lever_Center 좌표: ({cx}, {cy})")

                cv2.drawContours(rgb_img, [lever_contour], -1, (0, 255, 0), 2)
                cv2.circle(rgb_img, (cx, cy), 3, (0, 255, 0), -1)
            else:
                print("lever 중심 계산 불가능")   
        else:
            print("lever 감지 X")

         # RGB, Depth, Purple, Combined, Process Mask 시각화
        cv2.imshow('RGB', rgb_img)
        cv2.imshow('Depth', depth_mask)

        cv2.imshow('Purple_mask', purple_mask)
        cv2.imshow('Combined', combined_mask)
        cv2.imshow('Processed', processed_mask)
    
        # 'q' 키 눌러서 루프 나오기
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
        
finally:
    # Kinect, OpenCV 종료
    k4a.stop()
    cv2.destroyAllWindows()