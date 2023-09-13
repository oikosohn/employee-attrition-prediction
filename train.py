import time
from collections import Counter
import traceback

import joblib
import numpy as np
import pandas as pd

import lightgbm as lgbm
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import recall_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

import mlflow

def main():
    try:
        # mlflow 세션 시작
        mlflow.start_run()

        # 데이터 불러오기
        train = pd.read_csv('./data/train.csv')

        # 전처리 : 범주형 특성을 숫자로 변환
        object_columns = train.select_dtypes(include=['object']).columns
        object_columns = object_columns.tolist()
        train[object_columns] = OrdinalEncoder().fit_transform(train[object_columns])

        # 피처와 라벨 분할
        X_train = train.iloc[:, :-1]
        y_train = train.iloc[:, -1]

        # 데이터 증강
        smote = SMOTE(sampling_strategy=1, random_state=42) 
        X_sm, y_sm = smote.fit_resample(X_train, y_train)
        print(Counter(y_sm)), len(X_sm)

        # 학습
        lgbm_params = {'n_estimators': 407,
                        'num_rounds': 274,
                        'learning_rate': 0.1,
                        'num_leaves': 195,
                        'max_depth': 9,
                        'min_data_in_leaf': 46,
                        'lambda_l1': 0.01,
                        'lambda_l2': 0.6,
                        'min_gain_to_split': 1.42,
                        'bagging_fraction': 0.45,
                        'feature_fraction': 0.3}


        # 모델 학습
        model = lgbm.LGBMClassifier(**lgbm_params)
        model.fit(X_sm, y_sm)
        y_pred = model.predict(X_sm)

        score = recall_score(y_sm, y_pred, pos_label=1)
        cm = confusion_matrix(y_sm, y_pred)
        cr = classification_report(y_sm, y_pred)
        print(score)
        print(cm)    
        print(cr)

        ### mlflow 로깅 시작 ###

        # 증간된 데이터 저장 및 로깅
        # mlflow.log_artifact(local_path=csv_filename, artifact_path='data')

        # 모델 파라미터 로깅
        time_stamp = int(time.time())
        exp_name = 'lgbm_clf'
        run = mlflow.get_run(run_id)
        mlflow.set_tag("mlflow.runName", run_name)
        mlflow.set_experiment(exp_name)

        model_str = repr(model).split('(')[0]
        mlflow.log_param('model', model_str)
        for key, value in lgbm_params.items():
            mlflow.log_param(key, value)

        # 혼동 행렬을 로그로 저장
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                mlflow.log_metric(f'confusion_matrix_{i}_{j}', cm[i][j])

        # 모델 경로 지정
        model_path = f'./model/lgbm_clf_{time_stamp}.pkl'

        # 로컬에 모델 저장
        joblib.dump(model, model_path)

        # mlflow에 모델 저장
        mlflow.log_artifact(local_path=model_path) 

        ### mlflow 로깅 종료 ###

        # mlflow 세션 종료
        mlflow.end_run()

    # 예외 처리: 오류 발생 시 로그로 저장
    except Exception as e:
        mlflow.log_param('status', 'error')
        mlflow.log_param('error_message', str(e))
        mlflow.log_param('traceback', traceback.format_exc())
        mlflow.end_run()

        log_data = {
            'status': 'error',
            'error_message': str(e),
            'traceback': traceback.format_exc()
        }

        artifact_path = "outputs"
        with mlflow.get_artifact_uri(artifact_path) as artifact_uri:
            artifact_file_path = f"{artifact_uri}/mlflow_error_log_{time_stamp}.txt"
            with open(artifact_file_path, 'w') as file:
                for key, value in log_data.items():
                    file.write(f'{key}: {value}\n')

        print("Logged information saved to 'mlflow_error_log.txt'")

if __name__ == "__main__":
    main()