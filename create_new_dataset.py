"""
This program is designed to create a new dataset of images and labels from the preliminary dataset.
We will use the preliminary dataset to extract images and labels for the new dataset. This new dataset 
will be used for custom training and fine-tuning YOLOv5.
"""

import cv2
from pathlib import Path
import random

def capture_frame(video_path, frame_number, output_path):
    """
    Captures a specific frame from a video and saves it as an image.

    Parameters:
    - video_path (str): Path to the video file.
    - frame_number (int): Frame number to capture (1-based).
    - output_path (str): Path to save the captured image.
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return
    
    # Set the frame number (OpenCV uses 0-based indexing)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    
    # Read the frame
    ret, frame = cap.read()
    
    if not ret:
        print(f"Error: Could not read frame {frame_number} from {video_path}.")
        return
    
    # Save the frame as an image
    cv2.imwrite(str(output_path), frame)
    cap.release()
    
    print(f"Frame {frame_number} from {video_path} saved as {output_path}.")

def create_new_dataset(preliminary_dir, vs13_dir, new_dataset_dir):
    """
    Creates a new dataset by extracting frames and bounding box labels from the preliminary dataset.

    Parameters:
    - preliminary_dir (Path): Path to the preliminary dataset directory.
    - vs13_dir (Path): Path to the VS13 dataset directory.
    - new_dataset_dir (Path): Path to the output new dataset directory.
    """
    label_dir = new_dataset_dir / 'labels'
    img_dir = new_dataset_dir / 'images'
    
    # Create directories if they don't exist
    label_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    
    text_files = list(preliminary_dir.glob('*/*.txt'))
    
    for text_file in text_files:
        with open(text_file, 'r') as f:
            lines = f.readlines()
            
            random.seed(42)
            selected_lines = random.sample(lines, k=min(len(lines), 3))  # Select up to 3 frames randomly
        
        file_name = text_file.stem
        car_model = text_file.parent.stem
        video_path = vs13_dir / car_model / f'{file_name}.MP4'
        
        if video_path.exists():
            for line in selected_lines:
                frame_number = int(line.split()[0])
                bbox = list(map(float, line.split()[1:]))
                
                # Calculate bounding box coordinates
                xmin, ymin, xmax, ymax = bbox
                center_x = (xmin + xmax) / 2
                center_y = (ymin + ymax) / 2
                w = xmax - xmin
                h = ymax - ymin

                # Paths for label and image files
                txt_filepath = label_dir / f'{file_name}_{frame_number}.txt'
                image_path = img_dir / f'{file_name}_{frame_number}.jpg'
                
                # Capture frame
                capture_frame(video_path, frame_number, image_path)
                
                # Read the captured image to get dimensions
                image = cv2.imread(str(image_path))
                height, width, _ = image.shape

                # Normalize the bounding box coordinates
                norm_center_x = center_x / width
                norm_center_y = center_y / height
                norm_w = w / width
                norm_h = h / height

                bbox_normalized = [norm_center_x, norm_center_y, norm_w, norm_h]

                # Write normalized bounding box to label file
                with open(txt_filepath, 'w') as w:
                    w.write(f"0 {' '.join(map(str, bbox_normalized))}")
                
                print(f"Label saved: {txt_filepath}")
                print(f"Frame: {frame_number}, BBox: {bbox_normalized}")

if __name__ == '__main__':
    cur_dir = Path.cwd()
    
    # Directories
    preliminary_dir = cur_dir / 'preliminary_dataset'
    vs13_dir = cur_dir / 'VS13'
    new_dataset_dir = cur_dir / 'New_Dataset'
    
    create_new_dataset(preliminary_dir, vs13_dir, new_dataset_dir)
