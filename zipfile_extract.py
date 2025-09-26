import zipfile
import os
import shutil

zip_path = '/Users/mac/Mini_Project/Model_script.py/dataset.zip'
extract_path = '/Users/mac/Documents/breast-cancer-project'

# Delete the folder if it exists
if os.path.exists(extract_path):
    shutil.rmtree(extract_path)
    print(f"🗑️ Deleted existing folder: {extract_path}")

# Extract the zip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
    print(f"✅ Dataset extracted to: {extract_path}")

