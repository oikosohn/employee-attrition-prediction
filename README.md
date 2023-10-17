# employee-attrition-prediction

## 설치
```
# 가상환경 생성
python -m venv mlops 

# 가상환경 생성
mlops\Scripts\activate.bat

# 라이브러리 설치
pip install --upgrade pip
pip install -r requirements.txt
```

## 실행
```
# 모델 예측
streamlit run front.py --server.port=8501

# 모델 학습
python mlflow_examply.py

# 모델 학습 기록 확인
mlflow ui
```
