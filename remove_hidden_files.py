import os

dataset_path = '/Users/mac/Documents/breast-cancer-project'
deleted = 0

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.startswith("."):
            os.remove(os.path.join(root, file))
            deleted += 1

print(f"✅ Deleted {deleted} hidden files.")
