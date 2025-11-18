from fastapi import FastAPI, File, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import torch
import torch.nn as nn
import torchvision.transforms as tf
from PIL import Image
import io
import torch.nn.functional as F
from pathlib import Path
import pymysql
from typing import Optional
import logging
import sys

# ==========================================
# 1. 로깅(Monitoring) 설정 추가
# ==========================================
# 로그를 남길 포맷 설정
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler("server.log"),  # 파일에 기록 (server.log)
        logging.StreamHandler(sys.stdout),  # 터미널에 출력 (Docker logs 확인용)
    ],
)
logger = logging.getLogger(__name__)

app = FastAPI()

# 정적 파일 설정
app.mount("/static", StaticFiles(directory="static"), name="static")

# 템플릿 설정
templates = Jinja2Templates(directory="templates")

# MySQL 데이터베이스 설정
DB_CONFIG = {
    "host": "your-host",
    "user": "admin",
    "password": "your-password",
    "database": "fish_db",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
}


def get_marine_life_info(class_name: str) -> Optional[dict]:
    """데이터베이스에서 해양생물 정보 조회"""
    try:
        connection = pymysql.connect(**DB_CONFIG)
        with connection.cursor() as cursor:
            sql = "SELECT habitat, size, description FROM marine_life WHERE class_name = %s"
            cursor.execute(sql, (class_name,))
            result = cursor.fetchone()
        connection.close()
        return result
    except Exception as e:
        # [로깅] DB 에러 기록
        logger.error(f"DB Connection Error: {str(e)}")
        return None


# AquaCNN 모델 클래스
class AquaCNN(nn.Module):
    def __init__(self, num_classes):
        super(AquaCNN, self).__init__()

        # Stage 1: 224→112
        self.conv1_1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, padding="same"
        )
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding="same"
        )
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.mp1 = nn.MaxPool2d(2, 2)

        # Stage 2: 112→56
        self.conv2_1 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding="same"
        )
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding="same"
        )
        self.bn2_2 = nn.BatchNorm2d(128)
        self.mp2 = nn.MaxPool2d(2, 2)

        # Stage 3: 56→28
        self.conv3_1 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding="same"
        )
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding="same"
        )
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding="same"
        )
        self.bn3_3 = nn.BatchNorm2d(256)
        self.mp3 = nn.MaxPool2d(2, 2)

        # Stage 4: 28→14
        self.conv4_1 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding="same"
        )
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding="same"
        )
        self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding="same"
        )
        self.bn4_3 = nn.BatchNorm2d(512)
        self.mp4 = nn.MaxPool2d(2, 2)

        # Stage 5: 14→7
        self.conv5_1 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding="same"
        )
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding="same"
        )
        self.bn5_2 = nn.BatchNorm2d(512)
        self.mp5 = nn.MaxPool2d(2, 2)

        # Output
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.mp1(x)

        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = self.mp2(x)

        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = self.relu(self.bn3_3(self.conv3_3(x)))
        x = self.mp3(x)

        x = self.relu(self.bn4_1(self.conv4_1(x)))
        x = self.relu(self.bn4_2(self.conv4_2(x)))
        x = self.relu(self.bn4_3(self.conv4_3(x)))
        x = self.mp4(x)

        x = self.relu(self.bn5_1(self.conv5_1(x)))
        x = self.relu(self.bn5_2(self.conv5_2(x)))
        x = self.mp5(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# 모델 및 설정 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_class_names(data_path="./data/train"):
    train_path = Path(data_path)
    if not train_path.exists():
        logger.error(f"Data path not found: {data_path}")  # [로깅]
        raise FileNotFoundError(f"{data_path} 폴더를 찾을 수 없습니다!")

    class_names = sorted([d.name for d in train_path.iterdir() if d.is_dir()])

    if not class_names:
        logger.error(f"No classes found in {data_path}")  # [로깅]
        raise ValueError(f"{data_path}에 클래스 폴더가 없습니다!")

    return class_names


try:
    class_names = get_class_names()
    num_classes = len(class_names)
    logger.info(f"Classes loaded: {class_names}")  # [로깅] 로드 성공
except Exception as e:
    logger.error(f"Initialization Error: {e}")
    class_names = []
    num_classes = 0

# 모델 로드
model = AquaCNN(num_classes=num_classes)
try:
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully.")  # [로깅] 모델 로드 성공
except Exception as e:
    logger.critical(f"Failed to load model: {e}")  # [로깅] 모델 로드 실패 (심각)

# 이미지 전처리
image_transform = tf.Compose(
    [
        tf.Resize((224, 224)),
        tf.ToTensor(),
        tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 전처리
        image_tensor = image_transform(image).unsqueeze(0).to(device)

        # 예측
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class = class_names[predicted_idx.item()]
        confidence_percent = confidence.item() * 100

        # [로깅] 예측 결과 기록 (가장 중요한 부분!)
        log_msg = f"File: {file.filename} -> Prediction: {predicted_class} ({confidence_percent:.2f}%)"
        logger.info(log_msg)

        # 신뢰도 체크 (50% 미만이면 예외 처리)
        if confidence_percent < 50:
            # [로깅] 낮은 신뢰도 경고
            logger.warning(
                f"Low Confidence Alert! File: {file.filename}, Conf: {confidence_percent:.2f}%"
            )

            return JSONResponse(
                {
                    "success": False,
                    "is_low_confidence": True,
                    "confidence": f"{confidence_percent:.2f}%",
                    "message": "확인할 수 없는 이미지입니다. 더 명확한 이미지를 업로드해주세요.",
                    "filename": file.filename,
                }
            )

        # 데이터베이스에서 정보 조회
        marine_info = get_marine_life_info(predicted_class)

        if marine_info:
            logger.info(
                f"DB info retrieved for {predicted_class}"
            )  # [로깅] DB 조회 성공
            return JSONResponse(
                {
                    "success": True,
                    "predicted_class": predicted_class,
                    "confidence": f"{confidence_percent:.2f}%",
                    "filename": file.filename,
                    "info": {
                        "habitat": marine_info["habitat"],
                        "size": marine_info["size"],
                        "description": marine_info["description"],
                    },
                }
            )
        else:
            # DB에 정보가 없는 경우
            logger.warning(
                f"No DB info found for {predicted_class}"
            )  # [로깅] DB 조회 실패
            return JSONResponse(
                {
                    "success": True,
                    "predicted_class": predicted_class,
                    "confidence": f"{confidence_percent:.2f}%",
                    "filename": file.filename,
                    "info": None,
                    "message": "생물 정보를 찾을 수 없습니다.",
                }
            )

    except Exception as e:
        # [로깅] 처리 중 발생한 모든 에러 기록
        logger.error(f"Error processing image {file.filename}: {str(e)}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
