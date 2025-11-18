# 1. 베이스 이미지: 파이썬 3.9 버전을 사용합니다 (가볍고 안정적)
FROM python:3.9-slim

# 2. 작업 폴더 설정: 컨테이너 내부의 /app 폴더에서 작업을 수행합니다
WORKDIR /app

# 3. 필수 라이브러리 설치: 이미지 처리에 필요한 시스템 파일들을 설치합니다
# (OpenCV 등을 사용할 때 에러를 방지하기 위함)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 4. requirements.txt 먼저 복사해서 라이브러리 설치
# (코드가 바뀌어도 라이브러리 설치 단계는 캐싱되어 속도가 빨라집니다)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 나머지 모든 프로젝트 파일 복사
COPY . .

# 6. 포트 열기: FastAPI가 사용하는 8000번 포트
EXPOSE 8000

# 7. 실행 명령어: 서버를 켜는 명령어입니다
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]