# import torchvision.transforms as tf
# import os
# from PIL import Image
# import time


# def preprocess_dataset():
#     transform = tf.Compose([tf.Resize(224), tf.CenterCrop(224), tf.ToTensor()])
#     to_pil = tf.ToPILImage()

#     folers = [
#         "./data/jelly_fish_processed",
#         "./data/lobster_processed",
#         "./data/octopus_processed",
#         "./data/otter_processed",
#         "./data/puffer_processed",
#         "./data/sea_horse_processed",
#         "./data/shark_processed",
#         "./data/whale_processed",
#     ]
#     for i in folers:
#         os.makedirs(i, exist_ok=True)

#     data_folders = folers = [
#         "./data/Jelly fish",
#         "./data/Lobster",
#         "./data/Octopus",
#         "./data/Otter",
#         "./data/Puffer",
#         "./data/Sea Horse",
#         "./data/Shark",
#         "./data/Whale",
#     ]

#     for i in data_folders:
#         print(f"\n처리 중: {i}")

#         for file in os.listdir(i):
#             input_path = os.path.join(i, file)
#             img = Image.open(input_path)
#             original_size = img.size

#             tensor_img = transform(img)
#             print(f"Tensor shape: {tensor_img.shape}")

#             processed_img = to_pil(tensor_img)

#             if "Crab" in i:
#                 output_path = os.path.join("./data/crab_processed", file)
#             elif "Dolphin" in i:
#                 output_path = os.path.join("./data/dolphin_processed", file)

#             rgb_img = processed_img.convert("RGB")
#             rgb_img.save(output_path, quality=95)
#             print(f"✅ {file}: {original_size} → {processed_img.size}")

#             time.sleep(0.1)


# preprocess_dataset()
# print(f"\n전처리 완료")


import torchvision.transforms as tf
import os
from PIL import Image
import time


def preprocess_dataset():
    transform = tf.Compose([tf.Resize(224), tf.CenterCrop(224), tf.ToTensor()])
    to_pil = tf.ToPILImage()

    output_folders = [
        "./data/jelly_fish_processed",
        "./data/lobster_processed",
        "./data/octopus_processed",
        "./data/otter_processed",
        "./data/puffer_processed",
        "./data/sea_horse_processed",
        "./data/shark_processed",
        "./data/whale_processed",
    ]
    for folder in output_folders:
        os.makedirs(folder, exist_ok=True)

    data_folders = [
        "./data/Jelly fish",
        "./data/Lobster",
        "./data/Octopus",
        "./data/Otter",
        "./data/Puffer",
        "./data/Sea Horse",
        "./data/Shark",
        "./data/Whale",
    ]

    folder_mapping = {
        "Jelly fish": "jelly_fish_processed",
        "Lobster": "lobster_processed",
        "Octopus": "octopus_processed",
        "Otter": "otter_processed",
        "Puffer": "puffer_processed",
        "Sea Horse": "sea_horse_processed",
        "Shark": "shark_processed",
        "Whale": "whale_processed",
    }

    for folder_path in data_folders:
        folder_name = os.path.basename(folder_path)
        output_folder = f"./data/{folder_mapping[folder_name]}"

        print(f"\n처리 중: {folder_path}")
        files = [f for f in os.listdir(folder_path) if not f.startswith(".")]
        total_files = len(files)

        for idx, file in enumerate(files, 1):
            try:
                input_path = os.path.join(folder_path, file)
                img = Image.open(input_path)
                original_size = img.size

                tensor_img = transform(img)
                processed_img = to_pil(tensor_img)
                rgb_img = processed_img.convert("RGB")

                output_path = os.path.join(output_folder, file)
                rgb_img.save(output_path, quality=95)  # 품질 85로 조정

                print(
                    f"✅ [{idx}/{total_files}] {file}: {original_size} → {processed_img.size}"
                )

                # 온도 관리: 10장마다 1초 휴식
                if idx % 10 == 0:
                    print(f"⏸️  CPU 휴식 중...")
                    time.sleep(1)
                else:
                    time.sleep(0.05)  # 매 이미지마다 50ms 대기

            except Exception as e:
                print(f"❌ 오류 발생 ({file}): {e}")

    print(f"\n✨ 전처리 완료!")


if __name__ == "__main__":
    preprocess_dataset()
