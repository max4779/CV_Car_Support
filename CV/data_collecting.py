import cv2
import mediapipe as mp
import numpy as np

# Mediapipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 데이터 수집 함수 수정
def collect_data(label, output_file, camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    data = []
    
    print(f"'{label}' data collecting. Press ESC to exit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 영상 좌우 반전
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 21개의 랜드마크 추출
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                # 데이터 검증: 랜드마크 길이가 63인지 확인
                if len(landmarks) == 63:
                    data.append((landmarks, label))
        
        cv2.imshow("Data Collection", frame)
        
        # ESC로 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 데이터 저장
    if data:  # 데이터가 비어 있지 않은 경우에만 저장
        np.save(output_file, np.array(data, dtype=object))
        print(f"데이터 저장 완료: {output_file}")
    else:
        print("데이터가 비어 있습니다. 저장하지 않았습니다.")

# 호출 시 camera_index를 설정
collect_data("fist", "fist_data.npy", camera_index=1)  # 주먹 제스처 (두 번째 카메라 사용)
collect_data("open_hand", "open_hand_data.npy", camera_index=1)  # 손을 핀 상태 제스처 (두 번째 카메라 사용)
collect_data("index_finger", "index_finger_data.npy", camera_index=1)  # 검지 손가락을 펴는 제스처 (두 번째 카메라 사용)
collect_data("two_finger", "two_finger_data.npy", camera_index=1)  # 검지와 중지 손가락을 펴는 제스처 (두 번째 카메라 사용)
