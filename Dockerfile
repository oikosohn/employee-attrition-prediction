# 데비안 운영체제 사용
FROM python:3.11-slim

COPY . /app
WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

RUN apt-get update
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 가상환경 생성
# RUN python -m venv /venv
# RUN /venv/bin/pip install --upgrade pip
# RUN /venv/bin/pip install -r requirements.txt

CMD ["streamlit", "run", "front.py", "--server.port=9000"]
CMD ["python", "train.py"]
CMD ["mlflow", "server"]
