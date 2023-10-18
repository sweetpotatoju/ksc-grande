import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

pd_list = []
pf_list = []
bal_list = []
fir_list = []

def classifier_eval(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print('혼동행렬 : ', cm)
    PD = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    print('PD : ', PD)
    PF = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    print('PF : ', PF)
    balance = 1 - (((0 - PF) * (0 - PF) + (1 - PD) * (1 - PD)) / 2)
    print('balance : ', balance)
    FI = (cm[1, 1] + cm[0, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
    FIR = (PD - FI) / PD
    print('FIR : ', FIR)
    return PD, PF, balance, FIR

# CSV 파일 경로를 지정
csv_file_path ="EQ.csv"

# CSV 파일을 데이터프레임으로 읽어오기
df = pd.read_csv(csv_file_path)

# 데이터프레임에서 특징(X)과 목표 변수(y) 추출
X = df.drop(columns=['class'])
y = df['class']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# K-겹 교차 검증을 설정
k = 10  # K 값 설정
kf = KFold(n_splits=k, shuffle=True, random_state=42)

scaler = MinMaxScaler()

# K-겹 교차 검증 수행
for train_index, val_index in kf.split(X_train):
    X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

    # Min-Max 정규화 수행
    X_fold_train_normalized = scaler.fit_transform(X_fold_train)
    X_fold_val_normalized = scaler.transform(X_fold_val)
    X_test_normalized = scaler.transform(X_test)

    # SMOTE를 사용하여 학습 데이터 오버샘플링
    smote = SMOTE(random_state=42)
    X_fold_train_resampled, y_fold_train_resampled = smote.fit_resample(X_fold_train_normalized, y_fold_train)

    # 랜덤 포레스트 분류 모델 초기화 및 훈련
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    rf_classifier.fit(X_fold_train_resampled, y_fold_train_resampled)

    # 테스트 데이터에 대한 예측 수행
    rf_preds = rf_classifier.predict(X_test_normalized)
    print(rf_preds)

    # 분류 모델 평가 및 결과 기록
    PD, PF, balance, FIR = classifier_eval(y_test, rf_preds)
    pd_list.append(PD)
    pf_list.append(PF)
    bal_list.append(balance)
    fir_list.append(FIR)

# 결과 출력
print('평균 PD: {}'.format((sum(pd_list) / len(pd_list))))
print('평균 PF: {}'.format((sum(pf_list) / len(pf_list))))
print('평균 balance: {}'.format((sum(bal_list) / len(bal_list))))
print('평균 FIR: {}'.format((sum(fir_list) / len(fir_list))))
