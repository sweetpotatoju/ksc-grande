import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import KFold
import category_encoders as ce
import sklearn
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

from GRANDE import GRANDE

# CSV 파일 경로를 지정
csv_file_path = "EQ.csv"

# CSV 파일을 데이터프레임으로 읽어오기
df = pd.read_csv(csv_file_path)
# print(df)

# 데이터프레임에서 특징(X)과 목표 변수(y) 추출
X = df.drop(columns=['class'])
y = df['class']  # 'target' 열을 목표 변수로 사용


X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# K-겹 교차 검증을 설정합니다
k = 5  # K 값 (원하는 폴드 수) 설정
kf = KFold(n_splits=k, shuffle=True, random_state=42)

scaler = MinMaxScaler()
# K-겹 교차 검증 수행
for train_index, val_index in kf.split(X_train):
    X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

#전처리
    # Min-Max 정규화 수행(o)
    X_fold_train_normalized = scaler.fit_transform(X_fold_train)
    X_fold_val_normalized = scaler.transform(X_fold_val)


    # SMOTE를 사용하여 학습 데이터 오버샘플링
    smote = SMOTE(random_state=42)
    X_fold_train_resampled, y_fold_train_resampled = smote.fit_resample(X_fold_train_normalized, y_fold_train)




    params = {
        'depth': 5,
        'n_estimators': 2048,

        'learning_rate_weights': 0.005,
        'learning_rate_index': 0.01,
        'learning_rate_values': 0.01,
        'learning_rate_leaf': 0.01,

        'optimizer': 'SWA',
        'cosine_decay_steps': 0,

        'initializer': 'RandomNormal',

        'loss': 'crossentropy',
        'focal_loss': False,

        'from_logits': True,
        'apply_class_balancing': True,

        'dropout': 0.0,

        'selected_variables': 0.8,
        'data_subset_fraction': 1.0,
}

    args = {
    'epochs': 100,
    'early_stopping_epochs': 25,
    'batch_size': 64,

    'objective': 'binary',
    'evaluation_metrics': ['F1'], # F1, Accuracy, R2
    'random_seed': 42,
    'verbose': 1,
}

    model = GRANDE(params=params, args=args)

    model.fit(X_train= X_fold_train_resampled,
          y_train=y_fold_train_resampled,
          X_val=X_valid,
          y_val=y_valid)


preds = model.predict(X_test)

accuracy = sklearn.metrics.accuracy_score(y_test, np.round(preds[:,1]))
f1_score = sklearn.metrics.f1_score(y_test, np.round(preds[:,1]), average='macro')
roc_auc = sklearn.metrics.roc_auc_score(y_test, preds[:,1], average='macro')

print('Accuracy:', accuracy)
print('F1 Score:', f1_score)
print('ROC AUC:', roc_auc)
