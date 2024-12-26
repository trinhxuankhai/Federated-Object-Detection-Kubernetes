import os
import shutil
import random
import yaml
from math import ceil

def split_for_federated(base_dir, num_clients, seed=42):
    """
    Split training data across multiple clients for federated learning
    
    Args:
        base_dir (str): Base directory containing the dataset
        num_clients (int): Number of clients to split the data for
        seed (int): Random seed for reproducibility
    """
    random.seed(seed)
    
    # Create directory for federated setup
    fed_base_dir = f"{base_dir}_federated"
    os.makedirs(fed_base_dir, exist_ok=True)
    
    # Read original yaml file
    with open(os.path.join(base_dir, 'data.yaml'), 'r') as f:
        original_yaml = yaml.safe_load(f)
    
    # Get list of training images
    train_img_dir = os.path.join(base_dir, 'train', 'images')
    train_images = [f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Calculate images per client
    total_images = len(train_images)
    images_per_client = ceil(total_images / num_clients)
    
    # Shuffle images
    random.shuffle(train_images)
    
    # Create directories and split data for each client
    for client_id in range(num_clients):
        # Create client directory structure
        client_dir = os.path.join(fed_base_dir, f'client_{client_id}')
        os.makedirs(os.path.join(client_dir, 'train', 'images'), exist_ok=True)
        os.makedirs(os.path.join(client_dir, 'train', 'labels'), exist_ok=True)
        
        # Copy validation and test directories from original dataset
        for split in ['valid', 'test']:
            shutil.copytree(
                os.path.join(base_dir, split),
                os.path.join(client_dir, split),
                dirs_exist_ok=True
            )
        
        # Get this client's images
        start_idx = client_id * images_per_client
        end_idx = min((client_id + 1) * images_per_client, total_images)
        client_images = train_images[start_idx:end_idx]
        
        # Copy images and labels for this client
        for img_file in client_images:
            # Copy image
            shutil.copy2(
                os.path.join(base_dir, 'train', 'images', img_file),
                os.path.join(client_dir, 'train', 'images', img_file)
            )
            
            # Copy corresponding label
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(base_dir, 'train', 'labels', label_file)
            if os.path.exists(label_path):
                shutil.copy2(
                    label_path,
                    os.path.join(client_dir, 'train', 'labels', label_file)
                )
        
        # Create client's yaml file
        client_yaml = dict(original_yaml)  # Copy original yaml
        
        # Update paths for this client
        client_yaml['train'] = '../train/images'
        client_yaml['val'] = '../valid/images'
        client_yaml['test'] = '../test/images'
        
        # Add client-specific metadata
        client_yaml['client_id'] = client_id
        client_yaml['total_clients'] = num_clients
        client_yaml['client_train_images'] = len(client_images)
        
        # Save client's yaml
        with open(os.path.join(client_dir, 'data.yaml'), 'w') as f:
            yaml.dump(client_yaml, f, sort_keys=False)

    # Create a summary file
    summary = {
        'total_images': total_images,
        'num_clients': num_clients,
        'images_per_client': images_per_client,
        'seed': seed,
        'original_yaml': original_yaml
    }
    
    with open(os.path.join(fed_base_dir, 'federation_summary.yaml'), 'w') as f:
        yaml.dump(summary, f, sort_keys=False)

    return fed_base_dir

# Usage example
if __name__ == "__main__":
    base_directory = "./datasets/brain-tumor-detection/BrainTumorYolov8_subset"  # Use subset from previous script
    num_clients = 2  # Number of clients to create
    
    fed_dir = split_for_federated(base_directory, num_clients)
    print(f"Federated learning setup created at: {fed_dir}")