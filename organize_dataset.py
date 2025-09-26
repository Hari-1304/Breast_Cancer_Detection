import os
import shutil

source_dir = '/Users/mac/Documents/breast-cancer-project'
target_dir = '/Users/mac/Documents/breast-cancer-organized'

class_0_dir = os.path.join(target_dir, 'class_0')
class_1_dir = os.path.join(target_dir, 'class_1')
os.makedirs(class_0_dir, exist_ok=True)
os.makedirs(class_1_dir, exist_ok=True)

count_0 = 0
count_1 = 0

for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(root, file)
            if '/0' in root or '/class_0' in root:
                shutil.copy(full_path, os.path.join(class_0_dir, file))
                count_0 += 1
            elif '/1' in root or '/class_1' in root:
                shutil.copy(full_path, os.path.join(class_1_dir, file))
                count_1 += 1

print(f"✅ Moved {count_0} class_0 images and {count_1} class_1 images to '{target_dir}'.")

