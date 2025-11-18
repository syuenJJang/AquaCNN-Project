import os

# 데이터 폴더 경로 (본인 경로에 맞게 수정)
target_folder = "./data"

# data 폴더가 없으면 생성
os.makedirs(target_folder, exist_ok=True)

count = 0
# 하위 모든 폴더를 돌면서 확인
for root, dirs, files in os.walk(target_folder):
    for dir_name in dirs:
        folder_path = os.path.join(root, dir_name)
        gitkeep_path = os.path.join(folder_path, ".gitkeep")

        # 폴더 안에 파일이 하나도 없다면 .gitkeep 생성
        if not os.listdir(folder_path):
            with open(gitkeep_path, "w") as f:
                pass  # 빈 파일 생성
            print(f"생성됨: {gitkeep_path}")
            count += 1

if count == 0:
    print("이미 모든 폴더에 파일이 있거나, 빈 폴더가 없습니다.")
else:
    print(f"\n총 {count}개의 빈 폴더에 .gitkeep 파일을 만들었습니다!")
