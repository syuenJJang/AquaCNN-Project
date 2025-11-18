# 🐟 AquaCNN: Marine Life Classification AI

해양 생물 이미지를 업로드하면 CNN 모델을 통해 어떤 생물인지(Jelly fish, Lobster, Shark 등) 분류하고 정보를 제공하는 AI 웹 서비스입니다.

## 📌 주요 기능
- **이미지 분류**: PyTorch 기반 Custom CNN 모델(`AquaCNN`) 사용
- **웹 인터페이스**: FastAPI & Jinja2 템플릿을 활용한 반응형 웹 UI
- **결과 시각화**: 예측된 클래스와 신뢰도(Confidence) 표시
- **생물 정보 제공**: 예측 결과에 따른 생태 정보 출력

## 🛠 기술 스택 (Tech Stack)
- **Language**: Python 3.9+
- **AI/ML**: PyTorch, Torchvision, PIL
- **Web Backend**: FastAPI, Uvicorn
- **Frontend**: HTML5, CSS3, JavaScript
- **Infra/Ops**: Docker (예정)

## 🚀 설치 및 실행 방법 (Installation)

1. **레포지토리 클론**
   ```bash
   git clone [레포지토리 주소]
   cd [프로젝트 폴더]

의존성 설치
pip install -r requirements.txt

서버 실행
uvicorn main:app --reload