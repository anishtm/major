"""
The script train_yolov5.py adapted for a local environment with the base directory as New_Dataset.
The script will split the dataset into training and testing sets, copy the relevant files to separate directories,
generate text files listing the paths of these images, and create a YOLOv5 configuration file in YAML format, and train for 100 epochs.
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml
from ultralytics import YOLO

def generate_annot_list(image_paths):
    """
    Generate a list of annotation file paths corresponding to the image paths.

    Args:
        image_paths (list): List of image file paths.

    Returns:
        list: List of annotation file paths.
    """
    annot_list = []
    for path in image_paths:
        # Get the file name without extension
        file_name = path.stem
        # Construct label file path
        label_path = path.parent.parent / 'labels' / (file_name + '.txt')
        annot_list.append(label_path)
    return annot_list

def copy_files(file_list, destination_folder):
    """
    Copy files to a destination folder.

    Args:
        file_list (list): List of file paths.
        destination_folder (str): Path to the destination folder.
    """
    for file in file_list:
        # Get the full path of the file
        file_path = os.path.abspath(file)
        # Copy the file to the destination folder
        shutil.copy(file_path, destination_folder)

def generate_txt_files(paths, output_file):
    """
    Dump a list of paths to a text file.

    Args:
        paths (list): List of paths to dump.
        output_file (str): Path to the output text file.
    """
    with open(output_file, 'w') as file:
        for path in paths:
            file.write(str(path) + '\n')

if __name__ == '__main__':
    # Set the base directory
    base_dir = Path.cwd() / 'New_Dataset'
    img_dir = base_dir / 'images'
    label_dir = base_dir / 'labels'

    # Get the list of image and label files
    image_list = list(img_dir.glob('*.jpg'))
    label_list = list(label_dir.glob('*.txt'))

    print(f"Number of images: {len(image_list)}, Number of labels: {len(label_list)}")

    # Split the dataset into training and testing sets
    img_train, img_test = train_test_split(image_list, test_size=0.2, random_state=42)
    label_train = generate_annot_list(img_train)
    label_test = generate_annot_list(img_test)

    # Define the directory paths for the split datasets
    train_img_dir = base_dir / 'datasets' / 'img' / 'train'
    train_label_dir = base_dir / 'datasets' / 'labels' / 'train'
    test_img_dir = base_dir / 'datasets' / 'img' / 'test'
    test_label_dir = base_dir / 'datasets' / 'labels' / 'test'

    # Create directories if they don't exist
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)

    # Copy the image and label files to the respective directories
    copy_files(img_train, train_img_dir)
    copy_files(label_train, train_label_dir)
    copy_files(img_test, test_img_dir)
    copy_files(label_test, test_label_dir)

    # Generate text files listing the paths of the images for training and testing
    txt_dir = base_dir / 'texts'
    os.makedirs(txt_dir, exist_ok=True)

    generate_txt_files([str(p) for p in train_img_dir.glob('*.jpg')], txt_dir / 'train.txt')
    generate_txt_files([str(p) for p in test_img_dir.glob('*.jpg')], txt_dir / 'test.txt')

    # Create a YOLOv5 configuration file in YAML format
    config = {
        'path': str(txt_dir),
        'train': 'train.txt',
        'val': 'test.txt',
        'nc': 1,
        'names': {
            0: 'Car'
        }
    }

    config_path = base_dir / 'config.yaml'
    with open(config_path, 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)

    print("Training and testing datasets prepared and configuration file created.")

    # Load YOLOv5 model
    model = YOLO('yolov5m.pt')  # Use the appropriate path or model name

    # Train YOLOv5 model
    results = model.train(data=str(config_path), epochs=100)
    print("Training completed.")
