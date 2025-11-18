import os

folders_path = [
    "./data/Jelly fish",
    "./data/Lobster",
    "./data/Octopus",
    "./data/Otter",
    "./data/Puffer",
    "./data/Sea Horse",
    "./data/Shark",
    "./data/Whale",
]

for folder in folders_path:
    print(f"{folder} 폴더 작업 시작")
    count = 1
    new_ext = ".jpg"
    file_list = os.listdir(folder)
    for filename in file_list:
        old_path = os.path.join(folder, filename)
        if os.path.isfile(old_path):
            _, file_ext = os.path.splitext(filename)
            new_filename = f"{count}{new_ext}"
            new_path = os.path.join(folder, new_filename)
            os.rename(old_path, new_path)
            count += 1
print("작업 완료")
