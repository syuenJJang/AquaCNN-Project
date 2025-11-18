# import os
# import shutil


# def split_dataset():
#     folders = [
#         "./data/Jelly fish",
#         "./data/Lobster",
#         "./data/Octopus",
#         "./data/Otter",
#         "./data/Puffer",
#         "./data/Sea Horse",
#         "./data/Shark",
#         "./data/Whale",
#     ]

#     for folder in folders:
#         if not os.path.exists(folder):
#             os.makedirs(folder)

#     crab_images = []
#     dolphin_images = []

#     for file in os.listdir("./data/crab_processed"):
#         if file.lower().endswith("jpg"):
#             crab_images.append(file)

#     for file in os.listdir("./data/dolphin_processed"):
#         if file.lower().endswith("jpg"):
#             dolphin_images.append(file)

#     total_crab = len(crab_images)
#     train_crab_end = int(total_crab * 0.8)
#     val_crab_end = int(total_crab * 0.9)

#     print(
#         f"새우 이미지 분배: Train {train_crab_end}, Validate {val_crab_end - train_crab_end}, Test {total_crab - val_crab_end}"
#     )

#     crab_train_data = crab_images[0:train_crab_end]
#     for file in crab_train_data:
#         shutil.copy(f"./data/crab_processed/{file}", "./data/train/crab")

#     crab_validate_data = crab_images[train_crab_end:val_crab_end]
#     for file in crab_validate_data:
#         shutil.copy(f"./data/crab_processed/{file}", "./data/val/crab")

#     crab_test_data = crab_images[val_crab_end:]
#     for file in crab_test_data:
#         shutil.copy(f"./data/crab_processed/{file}", "./data/test/crab")

#     total_dolphin = len(dolphin_images)
#     train_dolphin_end = int(total_dolphin * 0.8)
#     val_dolphin_end = int(total_dolphin * 0.9)

#     print(
#         f"오징어 이미지 분배: Train {train_dolphin_end}, Validate {val_dolphin_end - train_dolphin_end}, Test {total_dolphin - val_dolphin_end}"
#     )

#     dolphin_train_data = dolphin_images[0:train_dolphin_end]
#     for file in dolphin_train_data:
#         shutil.copy(f"./data/dolphin_processed/{file}", "./data/train/dolphin")

#     dolphin_validate_data = dolphin_images[train_dolphin_end:val_dolphin_end]
#     for file in dolphin_validate_data:
#         shutil.copy(f"./data/dolphin_processed/{file}", "./data/val/dolphin")

#     dolphin_test_data = dolphin_images[val_dolphin_end:]
#     for file in dolphin_test_data:
#         shutil.copy(f"./data/dolphin_processed/{file}", "./data/test/dolphin")


# split_dataset()


import os
import shutil


def split_dataset():
    # 클래스별 폴더 정의
    classes = {
        "Jelly fish": "jelly_fish_processed",
        "Lobster": "lobster_processed",
        "Octopus": "octopus_processed",
        "Otter": "otter_processed",
        "Puffer": "puffer_processed",
        "Sea Horse": "sea_horse_processed",
        "Shark": "shark_processed",
        "Whale": "whale_processed",
    }

    # train/val/test 폴더 생성
    for split in ["train", "val", "test"]:
        for class_name in classes.keys():
            folder_path = f"./data/{split}/{class_name}"
            os.makedirs(folder_path, exist_ok=True)

    # 각 클래스별로 처리
    for class_name, processed_folder in classes.items():
        processed_path = f"./data/{processed_folder}"

        # 이미지 파일 목록 가져오기
        images = []
        for file in os.listdir(processed_path):
            if file.lower().endswith(("jpg", "jpeg", "png")):
                images.append(file)

        # 데이터 분할 (80% train, 10% val, 10% test)
        total = len(images)
        train_end = int(total * 0.8)
        val_end = int(total * 0.9)

        print(f"\n{class_name} 이미지 분배:")
        print(f"  Train: {train_end}장")
        print(f"  Val: {val_end - train_end}장")
        print(f"  Test: {total - val_end}장")

        # Train 데이터 복사
        train_data = images[0:train_end]
        for file in train_data:
            src = os.path.join(processed_path, file)
            dst = f"./data/train/{class_name}/{file}"
            shutil.copy(src, dst)

        # Validation 데이터 복사
        val_data = images[train_end:val_end]
        for file in val_data:
            src = os.path.join(processed_path, file)
            dst = f"./data/val/{class_name}/{file}"
            shutil.copy(src, dst)

        # Test 데이터 복사
        test_data = images[val_end:]
        for file in test_data:
            src = os.path.join(processed_path, file)
            dst = f"./data/test/{class_name}/{file}"
            shutil.copy(src, dst)

    print("\n✨ 데이터셋 분할 완료!")


if __name__ == "__main__":
    split_dataset()
