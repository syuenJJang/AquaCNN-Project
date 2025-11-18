import os

# 파일 이름을 다시 정렬할 폴더 목록
processed_folders = ["./data/sea_turtle_processed", "./data/seal_processed"]

for folder in processed_folders:
    print(f"\n--- {folder} 폴더 작업 시작 ---")

    try:
        # --- 0. 파일 목록을 숫자로 정렬 ---

        # '10.jpg' -> 10 으로 바꿔주는 함수
        def get_numeric_key(filename):
            name_part = os.path.splitext(filename)[0]
            try:
                return int(name_part)
            except ValueError:
                # 숫자가 아닌 파일(예: .DS_Store)은 무시
                return -1

        # 폴더 내 파일 목록을 가져와서
        file_list = []
        for f in os.listdir(folder):
            if os.path.isfile(os.path.join(folder, f)):
                file_list.append(f)

        # 숫자를 기준으로 정렬 (알파벳순 '1.jpg', '10.jpg', '2.jpg' 가 아닌
        # 숫자순 '1.jpg', '2.jpg', '10.jpg' 로 정렬됨)
        sorted_files = sorted(file_list, key=get_numeric_key)

        # 숫자가 아닌 파일(-1)은 리스트에서 제거
        sorted_files = [f for f in sorted_files if get_numeric_key(f) >= 0]

        if not sorted_files:
            print("처리할 숫자 파일이 없습니다. 건너뜁니다.")
            continue

        # --- 1단계: 모든 파일을 임시 이름(.temp)으로 변경 ---
        # (이 작업을 먼저 해야 이름 충돌(FileExistsError)이 안 생김)

        temp_paths = []  # 임시 파일 경로를 순서대로 저장할 리스트

        for filename in sorted_files:
            old_path = os.path.join(folder, filename)
            temp_name = f"{filename}.temp"  # 예: "120.jpg.temp"
            temp_path = os.path.join(folder, temp_name)

            os.rename(old_path, temp_path)
            temp_paths.append(temp_path)  # "순서대로" 임시 경로 저장

        print(f"{len(temp_paths)}개 파일을 임시 이름으로 변경했습니다.")

        # --- 2단계: 임시 파일들을 최종 순서(1, 2, 3...)로 변경 ---
        count = 1
        for temp_path in temp_paths:  # 정렬된 순서 그대로 반복

            # 임시 이름(예: 120.jpg.temp)에서 원본 확장자(예: .jpg) 찾기
            temp_filename = os.path.basename(temp_path)
            original_filename = temp_filename.replace(".temp", "")  # "120.jpg"
            _, file_ext = os.path.splitext(original_filename)  # ".jpg"

            # 새 파일 이름과 경로 생성
            final_filename = f"{count}{file_ext}"  # "119.jpg" (count가 119일 때)
            final_path = os.path.join(folder, final_filename)

            # 임시 파일 -> 최종 파일로 이름 변경
            os.rename(temp_path, final_path)
            print(f"변경: {temp_filename}  ->  {final_filename}")

            count += 1

    except FileNotFoundError:
        print(f"오류: '{folder}' 폴더를 찾을 수 없습니다.")
    except Exception as e:
        print(f"알 수 없는 오류 발생: {e}")

print("\n--- 모든 파일 숫자 순서 맞추기 완료 ---")
