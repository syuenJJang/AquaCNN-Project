import os

folder_path = [
    "./data/Jelly fish",
    "./data/Lobster",
    "./data/Octopus",
    "./data/Otter",
    "./data/Puffer",
    "./data/Sea Horse",
    "./data/Shark",
    "./data/Whale",
]

for folder_path in folder_path:
    file_count = len(os.listdir(folder_path))
    print(f"{folder_path}파일 개수: {file_count}")
