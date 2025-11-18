from PIL import Image
import torch.nn.functional as F
import torch
import training
import torchvision.transforms as tf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def predict_image(model, image_path, device, transform, class_names):
    model.eval()

    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"\n오류: '{image_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
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


if __name__ == "__main__":
    IMAGE_TO_PREDICT = "./seal_test_img.jpg"


print(f"\n\n이제 '{IMAGE_TO_PREDICT}' 파일로 최종 예측을 시작합니다...")

prediction_model = training.AquaCNN(num_classes=num_classes)
try:
    prediction_model.load_state_dict(torch.load("best_model.pth", map_location=device))
    prediction_model.to(device)

    class_names = training.train_dataset.classes
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
