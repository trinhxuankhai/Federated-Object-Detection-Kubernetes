import kagglehub
import subprocess
import os

# Download the latest version of the dataset
path = kagglehub.dataset_download("pkdarabi/medical-image-dataset-brain-tumor-detection", force_download=True)

# Specify the target folder
target_folder = "./datasets/brain-tumor-detection"

# Create the target folder if it doesn't exist
os.makedirs(target_folder, exist_ok=True)

# Use the `cp -a` command to copy the entire directory including all files and subdirectories
subprocess.run(["cp", "-a", f"{path}/.", target_folder], check=True)

print("All files copied to:", target_folder)
