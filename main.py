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

app = FastAPI()

# 정적 파일 설정
app.mount("/static", StaticFiles(directory="static"), name="static")

# 템플릿 설정
templates = Jinja2Templates(directory="templates")


# AquaCNN 모델 클래스 (기존 코드에서 가져오기)
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
        # Stage 1
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.mp1(x)

        # Stage 2
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = self.mp2(x)

        # Stage 3
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = self.relu(self.bn3_3(self.conv3_3(x)))
        x = self.mp3(x)

        # Stage 4
        x = self.relu(self.bn4_1(self.conv4_1(x)))
        x = self.relu(self.bn4_2(self.conv4_2(x)))
        x = self.relu(self.bn4_3(self.conv4_3(x)))
        x = self.mp4(x)

        # Stage 5
        x = self.relu(self.bn5_1(self.conv5_1(x)))
        x = self.relu(self.bn5_2(self.conv5_2(x)))
        x = self.mp5(x)

        # Output
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


# 모델 및 설정 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 새로운 코드


def get_class_names(data_path="./data/train"):
    train_path = Path(data_path)
    if not train_path.exists():
        raise FileNotFoundError(f"{data_path} 폴더를 찾을 수 없습니다!")

    class_names = sorted([d.name for d in train_path.iterdir() if d.is_dir()])

    if not class_names:
        raise ValueError(f"{data_path}에 클래스 폴더가 없습니다!")

    return class_names


class_names = get_class_names()
num_classes = len(class_names)

print(f"자동으로 로드된 클래스: {class_names}")
num_classes = len(class_names)

# 모델 로드
model = AquaCNN(num_classes=num_classes)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

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

        # 신뢰도 체크 (90% 미만이면 예외 처리)
        if confidence_percent < 90:
            return JSONResponse(
                {
                    "success": False,
                    "is_low_confidence": True,
                    "confidence": f"{confidence_percent:.2f}%",
                    "message": "확인할 수 없는 이미지입니다. 더 명확한 이미지를 업로드해주세요.",
                    "filename": file.filename,
                }
            )

        return JSONResponse(
            {
                "success": True,
                "predicted_class": predicted_class,
                "confidence": f"{confidence_percent:.2f}%",
                "filename": file.filename,
            }
        )

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
