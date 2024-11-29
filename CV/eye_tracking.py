import cv2
import mediapipe as mp
import time

# Mediapipe 설정
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
drawing_utils = mp.solutions.drawing_utils

last_direction = None
direction_start_time = 0

cap = cv2.VideoCapture(0) # 이거는 외부 웹캠에 연결하는 것이라서 내부 카메라 사용을 원하면 0으로

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            drawing_utils.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

            # 특정 랜드마크 예: 왼쪽 눈 (예: 33, 133), 오른쪽 눈 (예: 362, 263)
            left_cheek = face_landmarks.landmark[234]  # 왼쪽 얼굴
            right_cheek = face_landmarks.landmark[454] 
            nose_tip = face_landmarks.landmark[1]  # 코 끝
            chin = face_landmarks.landmark[152]   # 턱 끝

            # 시선 방향 계산
            if nose_tip.x < (left_cheek.x + right_cheek.x) / 2:
                current_direction = "오른쪽 주시 중"
            else:
                current_direction = "왼쪽 주시 중"

            # 시선 방향 유지 시간 계산
            if last_direction == current_direction:
                if time.time() - direction_start_time > 3:
                    print(f"{current_direction}")
            else:
                last_direction = current_direction
                direction_start_time = time.time()

            vertical_distance = chin.y - nose_tip.y
            # 고개가 숙였는지 판단
            if vertical_distance < 0.2:  # 기준값은 실험적으로 조정 필요
                print("고개를 숙임")
            

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
