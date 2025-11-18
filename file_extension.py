import os

folder_paths = [
    "./data/Jelly fish",
    "./data/Lobster",
    "./data/Octopus",
    "./data/Otter",
    "./data/Puffer",
    "./data/Sea Horse",
    "./data/Shark",
    "./data/Whale",
]

for folder_path in folder_paths:  #
    print(f"\n=== {folder_path} ===")
    extensions = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            _, extension = os.path.splitext(filename)
            extensions.append(extension.lower())

    print(set(extensions))
