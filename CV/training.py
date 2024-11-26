from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np
import joblib

# 데이터 로드
fist_data = np.load("data/fist_data.npy", allow_pickle=True)
open_hand_data = np.load("data/open_hand_data.npy", allow_pickle=True)
index_finger_data = np.load("data/index_finger_data.npy", allow_pickle=True)
two_finger_data = np.load("data/two_finger_data.npy", allow_pickle=True)
call_sign_data = np.load("data/call_sign_data.npy", allow_pickle=True)
music_sign_data = np.load("data/music_sign_data.npy", allow_pickle=True)

# 데이터 결합
data = np.concatenate((fist_data, open_hand_data, index_finger_data, two_finger_data, call_sign_data, music_sign_data), axis=0)

# 랜드마크(X)와 레이블(y) 추출
X = np.array([item[0] for item in data])  # 랜드마크
y = np.array([item[1] for item in data])  # 레이블

# 데이터 분할 (훈련:테스트 = 80%:20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM 모델 학습
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# 모델 평가
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 모델 저장
joblib.dump(model, "gesture_model.pkl")
print("모델 저장 완료: gesture_model.pkl")
