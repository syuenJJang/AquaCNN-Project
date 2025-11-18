import torch
import torchvision.transforms as tf
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn


train_transform = tf.Compose(
    [
        tf.ToTensor(),
        tf.Normalize(mean=[0.485, 0.486, 0.406], std=[0.229, 0.224, 0.225]),
        tf.RandomHorizontalFlip(p=0.5),
    ]
)

val_transform = tf.Compose(
    [
        tf.ToTensor(),
        tf.Normalize(mean=[0.485, 0.486, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class AquaCNN(nn.Module):
    def __init__(self, num_classes):
        super(AquaCNN, self).__init__()

        # ========== Stage 1: 224→112 ==========
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

        # ========== Stage 2: 112→56 ==========
        self.conv2_1 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding="same"
        )
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding="same"
        )
        self.bn2_2 = nn.BatchNorm2d(128)
        self.mp2 = nn.MaxPool2d(2, 2)

        # ========== Stage 3: 56→28 ==========
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

        # ========== Stage 4: 28→14 ==========
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

        # ========== Stage 5: 14→7 (새로운 스테이지!) ==========
        self.conv5_1 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding="same"
        )
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding="same"
        )
        self.bn5_2 = nn.BatchNorm2d(512)
        self.mp5 = nn.MaxPool2d(2, 2)

        # ========== Output ==========
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Stage 1: 224→112
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.mp1(x)

        # Stage 2: 112→56
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = self.mp2(x)

        # Stage 3: 56→28
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = self.relu(self.bn3_3(self.conv3_3(x)))
        x = self.mp3(x)

        # Stage 4: 28→14
        x = self.relu(self.bn4_1(self.conv4_1(x)))
        x = self.relu(self.bn4_2(self.conv4_2(x)))
        x = self.relu(self.bn4_3(self.conv4_3(x)))
        x = self.mp4(x)

        # Stage 5: 14→7
        x = self.relu(self.bn5_1(self.conv5_1(x)))
        x = self.relu(self.bn5_2(self.conv5_2(x)))
        x = self.mp5(x)

        # Output
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    Accuracy = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss = running_loss + loss.item()
        _, predicted = outputs.max(1)
        total = total + labels.size(0)
        Accuracy = Accuracy + predicted.eq(labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f"Batch: [{i+1}/{len(train_loader)})], Loss: {loss.item():.4f}")

    return running_loss / (len(train_loader)), 100.0 * Accuracy / total


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    Accuracy = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            Accuracy += predicted.eq(labels).sum().item()

    return running_loss / len(val_loader), 100.0 * Accuracy / total


# ==================================================================
# 4. (적용된 함수) 전체 예측 분석 함수
# ==================================================================
def log_all_predictions(model, loader, device, idx_to_class, show_details=False):
    """
    전체 데이터셋의 모든 샘플에 대한 예측 결과를 분석하는 함수
    """
    print("\n--- 전체 샘플 예측 결과 분석 ---")
    model.eval()
    total_samples = 0
    correct_samples = 0
    class_stats = {}

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            # 배치별 처리
            for j in range(len(images)):
                true_class = idx_to_class[labels[j].item()]
                pred_class = idx_to_class[predicted[j].item()]
                is_correct = true_class == pred_class

                # 처음 몇 개만 출력 (선택사항)
                if show_details and total_samples < 20:
                    correct_mark = "✓" if is_correct else "✗"
                    print(f"{correct_mark} 실제: {true_class:<15} | 예측: {pred_class}")

                # 통계 수집
                if is_correct:
                    correct_samples += 1
                total_samples += 1

                # 클래스별 통계
                if true_class not in class_stats:
                    class_stats[true_class] = {"correct": 0, "total": 0}
                class_stats[true_class]["total"] += 1
                if is_correct:
                    class_stats[true_class]["correct"] += 1

            # 진행상황 출력 (옵션)
            if (batch_idx + 1) % 10 == 0:
                print(f"처리 중... {total_samples}개 완료")

    # 전체 결과 출력
    overall_accuracy = (
        (correct_samples / total_samples) * 100 if total_samples > 0 else 0
    )

    print(f"\n--- 전체 샘플 분석 결과 ---")
    print(f"총 샘플 수: {total_samples}개")
    print(f"정답 샘플: {correct_samples}개")
    print(f"오답 샘플: {total_samples - correct_samples}개")
    print(f"**전체 샘플 정확도: {overall_accuracy:.2f}%**")

    print(f"\n--- 클래스별 상세 성능 ---")
    # 클래스 이름을 기준으로 정렬하여 출력
    for class_name in sorted(class_stats.keys()):
        stats = class_stats[class_name]
        class_acc = (
            (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        )
        print(
            f"{class_name:<15}: {stats['correct']}/{stats['total']} = {class_acc:.1f}%"
        )

    return {
        "total_samples": total_samples,
        "correct_samples": correct_samples,
        "accuracy": overall_accuracy,
        "class_stats": class_stats,
    }


# ========== 메인 실행 부분 (중요!) ==========
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 데이터셋 로드
    train_dataset = datasets.ImageFolder("./data/train/", transform=train_transform)
    val_dataset = datasets.ImageFolder("./data/val/", transform=val_transform)
    test_dataset = datasets.ImageFolder("./data/test/", transform=val_transform)

    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=6, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=6, pin_memory=True
    )

    print(f"Classes: {train_dataset.classes}")
    print(f"Class to index: {train_dataset.class_to_idx}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    num_classes = len(train_dataset.classes)

    # 모델, 손실함수, 옵티마이저
    model = AquaCNN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)

    # 학습
    epochs = 60
    best_train_acc = 0.0

    # for epoch in range(epochs):
    #     train_loss, train_acc = train_epoch(
    #         model, train_loader, criterion, optimizer, device
    #     )
    #     val_loss, val_acc = validate(model, val_loader, criterion, device)

    #     print(f"Epoch [{epoch+1}/{epochs}]")
    #     print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    #     print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    #     if train_acc > best_train_acc:
    #         best_train_acc = train_acc
    #         torch.save(model.state_dict(), "best_model.pth")
    #         print(f"Model saved based on best train accuracy: {train_acc:.2f}%")

    #     print("-" * 50)

    # log_all_predictions에 필요한 idx_to_class 맵 생성
    # idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    # ==================================================================
    # 최종 모델 상세 평가
    # ==================================================================
    # print("\n--- 최종 테스트 평가 시작 ---")
    # print("가장 성능이 좋았던 'best_model.pth'를 불러옵니다.")

    # # 'best_model.pth' 파일이 존재하는지 확인
    # try:
    #     # 모델에 저장된 가중치(state_dict)를 불러옵니다.
    #     # (주의: 모델 구조(AquaCNN)는 동일해야 함)
    #     model.load_state_dict(torch.load("best_model.pth"))
    #     model = model.to(device)  # device로 다시 보내기

    #     # log_all_predictions 함수를 test_loader에 대해 호출
    #     test_stats = log_all_predictions(
    #         model=model,
    #         loader=test_loader,  # 테스트 데이터로 최종 평가
    #         device=device,
    #         idx_to_class=idx_to_class,
    #         show_details=True,  # 처음 20개 샘플의 예측 결과도 함께 표시
    #     )
    #     print("\n최종 테스트 상세 분석 완료.")

    # except FileNotFoundError:
    #     print("\n[오류] 'best_model.pth' 파일을 찾을 수 없습니다.")
    #     print("먼저 위 학습 루프의 주석을 해제하고 모델을 학습시켜주세요.")
    # except Exception as e:
    #     print(f"\n모델 로드 또는 평가 중 오류 발생: {e}")

    """
    외부 이미지 테스트
    """

    from PIL import Image
    import torch.nn.functional as F

    def predict_image(model, image_path, device, transform, class_names):
        model.eval()

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(
                f"\n오류: '{image_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요."
            )
            return

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class = class_names[predicted_idx.item()]
        confidence_percent = confidence.item() * 100

        print("\n--- 단일 이미지 예측 결과 ---")
        print(f"입력 이미지: {image_path}")
        print(f"모델 예측: '{predicted_class}'")
        print(f"신뢰도: {confidence_percent:.2f}%")
        print("-----------------------------")


IMAGE_TO_PREDICT = "./dolphin_test_img.jpg"

print(f"\n\n이제 '{IMAGE_TO_PREDICT}' 파일로 최종 예측을 시작합니다...")

prediction_model = AquaCNN(num_classes=num_classes)
try:
    prediction_model.load_state_dict(torch.load("best_model.pth", map_location=device))
    prediction_model.to(device)

    class_names = train_dataset.classes
    print(f"자동으로 로드된 클래스: {class_names}")

    image_transform = tf.Compose(
        [
            tf.Resize((224, 224)),
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 예측 함수 호출
    predict_image(
        prediction_model, IMAGE_TO_PREDICT, device, image_transform, class_names
    )

except FileNotFoundError:
    print(f"\n오류: 모델 파일 'best_model.pth'을 찾을 수 없습니다.")
    print("모델 훈련을 먼저 실행하여 모델 파일을 생성해야 합니다.")
