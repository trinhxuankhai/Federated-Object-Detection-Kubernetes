import os
import shutil
import random
import yaml

def create_subset(source_dir, dest_dir, percentage=0.1):
    """
    Create a subset of a YOLOv8 dataset
    
    Args:
        source_dir (str): Path to original dataset
        dest_dir (str): Path where subset will be created
        percentage (float): Percentage of data to include in subset (0-1)
    """
    # Create destination directory structure
    splits = ['train', 'valid', 'test']
    for split in splits:
        os.makedirs(os.path.join(dest_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, split, 'labels'), exist_ok=True)

    # Process each split
    for split in splits:
        # Get list of all images
        images_dir = os.path.join(source_dir, split, 'images')
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Calculate number of files for subset
        num_files = int(len(image_files) * percentage)
        
        # Randomly select files
        selected_files = random.sample(image_files, num_files)
        
        # Copy selected files and their corresponding labels
        for filename in selected_files:
            # Copy image
            src_img = os.path.join(source_dir, split, 'images', filename)
            dst_img = os.path.join(dest_dir, split, 'images', filename)
            shutil.copy2(src_img, dst_img)
            
            # Copy corresponding label file
            label_filename = os.path.splitext(filename)[0] + '.txt'
            src_label = os.path.join(source_dir, split, 'labels', label_filename)
            dst_label = os.path.join(dest_dir, split, 'labels', label_filename)
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)

    # Copy and modify data.yaml
    src_yaml = os.path.join(source_dir, 'data.yaml')
    dst_yaml = os.path.join(dest_dir, 'data.yaml')
    
    if os.path.exists(src_yaml):
        with open(src_yaml, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Update paths in yaml
        # yaml_data['train'] = os.path.join(dest_dir, 'train')
        # yaml_data['val'] = os.path.join(dest_dir, 'valid')
        # yaml_data['test'] = os.path.join(dest_dir, 'test')
        
        with open(dst_yaml, 'w') as f:
            yaml.dump(yaml_data, f)

# Usage
source_directory = "./datasets/brain-tumor-detection/BrainTumorYolov8"
destination_directory = "./datasets/brain-tumor-detection/BrainTumorYolov8_subset"
random.seed(42)  # For reproducibility

create_subset(source_directory, destination_directory, 1)