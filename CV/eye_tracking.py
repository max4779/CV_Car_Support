import cv2
import mediapipe as mp

# Mediapipe 설정
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
drawing_utils = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1) # 이거는 외부 웹캠에 연결하는 것이라서 내부 카메라 사용을 원하면 0으로

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
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[362]

            # 시선 방향 계산
            eye_center = ((left_eye.x + right_eye.x) / 2, (left_eye.y + right_eye.y) / 2)
            if eye_center[0] < 0.5:  # 화면 오른쪽을 보고 있다면
                print("오른쪽 주시 중")
            elif eye_center[0] > 0.6:  # 화면 왼쪽을 보고 있다면
                print("왼쪽 주시 중")
            else:
                print("정면 주시 중")

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
