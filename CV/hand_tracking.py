import cv2
import mediapipe as mp
import joblib
import numpy as np
import time  # 시간 지연을 위한 라이브러리

# Mediapipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 모델 로드
model = joblib.load("model/gesture_model.pkl")

# 기능 상태
music_on = False
aircon_on=False
calling=False
last_music_on_state = False  # 이전 음악 상태 추적

# 제스처 감지
def detect_gesture():
    global music_on, last_music_on_state, aircon_on, calling
    cap = cv2.VideoCapture(1)  # 두 번째 카메라 사용 이거는 외부 웹캠에 연결하는 것이라서 내부 카메라 사용을 원하면 0으로
    last_gesture_time = 0  # 마지막 제스처 인식 시간
    gesture_delay = 1.5  # 제스처 인식 간격 (3초)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                # 제스처 예측
                prediction = model.predict([landmarks])
                gesture = prediction[0]
                
                # 현재 시간
                current_time = time.time()

                # 제스처가 인식된 후 일정 시간(3초) 이상 지난 경우에만 반응
                if current_time - last_gesture_time > gesture_delay:
                    last_gesture_time = current_time  # 마지막 제스처 인식 시간 갱신
                    
                    # 주먹은 사용X
                    # if gesture == "fist":
                    #     print("fist")
                    #     if not aircon_on: 
                    #         aircon_on=True
                    #         print("에어컨 ON")
                    #     else:
                    #         aircon_on=False
                    #         print("에어컨 OFF")
                        
                            
                    if gesture == "open_hand":  # 손을 핀 상태는 노래 재생
                        if not aircon_on: 
                            aircon_on=True
                            print("에어컨 ON")
                        else:
                            aircon_on=False
                            print("에어컨 OFF")
                        

                    elif gesture == "index_finger":  # 검지 손가락은 다음 곡으로 넘어감
                        if music_on:
                            print("다음 곡으로 넘어감!")
                        elif aircon_on:
                            print("온도 낮춤")
                        

                    elif gesture == "two_finger":  # 
                        if music_on:
                            print("이전 곡으로 넘어감!")
                        elif aircon_on:
                            print("온도 높임")
                        

                    elif gesture == "call_sign":  # 
                        if not calling:
                            calling = True
                            print("전화 받기")
                        else: 
                            calling=False
                            print("전화 끊기")

                    elif gesture == "music_sign":  # 
                        if not music_on:  # 음악이 꺼져 있으면 켬
                            music_on = True
                            print("음악 ON")
                        else: 
                            music_on=False
                            print("음악 OFF")

        cv2.imshow("Gesture Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 실행
detect_gesture()
