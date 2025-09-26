import os

dataset_path = '/Users/mac/Documents/breast-cancer-project'
image_extensions = ('.jpg', '.jpeg', '.png')
image_count = 0

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(image_extensions):
            image_count += 1

print(f"✅ Total image files remaining: {image_count}")
