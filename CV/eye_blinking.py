import cv2
import dlib
from scipy.spatial import distance as dist
import numpy as np
import time

## EAR 함수 정의
def eye_aspect_ratio(eye):
    # 눈의 수직 거리
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # 눈의 수평 거리
    C = dist.euclidean(eye[0], eye[3])
    # 눈 비율 계산
    ear = (A + B) / (2.0 * C)
    return ear


# EAR 임계값과 설정값
EYE_AR_THRESH = 0.25
WARNING_TIME = 2  # EAR 기준 경고 시간 (초)

# 타이머 초기화
last_blink_time = None

# dlib의 얼굴 검출기와 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./model/shape_predictor_68_face_landmarks.dat")

# 눈 좌표 (68개 점 중 눈에 해당하는 부분)
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# 비디오 스트림 시작
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 프레임 전처리
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    # 현재 경고 상태 초기화
    drowsy_warning = False

    for rect in rects:
        # 얼굴 랜드마크 찾기
        shape = predictor(gray, rect)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        
        # 왼쪽, 오른쪽 눈의 좌표
        leftEye = np.array(shape[lStart:lEnd])
        rightEye = np.array(shape[rStart:rEnd])

        # EAR 계산
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # 눈 주위 다각형 그리기
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # EAR 값이 임계값보다 낮으면
        if ear < EYE_AR_THRESH:
            if last_blink_time is None:
                last_blink_time = time.time()  # 처음 감긴 시간 기록
            elif time.time() - last_blink_time > WARNING_TIME:  # 2초 이상 유지
                drowsy_warning = True  # 졸음 경고 활성화
        else:
            last_blink_time = None  # EAR이 다시 올라가면 초기화

        # EAR 값 화면에 표시
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 경고 메시지 출력
    if drowsy_warning:
        cv2.putText(frame, "DROWSINESS ALERT!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # 결과 출력
    cv2.imshow("Frame", frame)

    # esc 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == 27:  # 27은 esc 키의 ASCII 코드
        break

cap.release()
cv2.destroyAllWindows()
