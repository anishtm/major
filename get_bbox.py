'''This script get_bbox.py is for finding the bounding boxes of the cars in the video. 
It takes videos from VS13 dataset and annotation txts from it, the 
annotation text contains speed in km/hr and the last time the vehicle appears in the video.'''

import cv2
from ultralytics import YOLO
from pathlib import Path
import os

def detect_and_save_bounding_boxes(video_path, output_path, time=None):
    """
    Detects cars in a video, saves the bounding box coordinates to a text file.

    Args:
        video_path (str): Path to the video file.
        output_path (str): Path to save the bounding box coordinates text file.
        time (float, optional): Last time the vehicle appears in the video (in seconds).
    """
    # Initialize YOLOv5 model
    model = YOLO('./models/best.pt')

    # Extract file name without extension
    filename = Path(video_path).stem
    print(f'Processing {filename}')
    
    # Initialize variables
    frame_number = 0
    cap = cv2.VideoCapture(str(video_path))

    # Calculate frame number to stop
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_to_stop = int(time * fps) + 1 if time else None
    
    # Open output file for writing
    with open(output_path, 'w') as f:
        while True:
            ret, frame = cap.read()
            if not ret or (frame_to_stop is not None and frame_number >= frame_to_stop):
                break
            
            # Perform object detection on the frame
            results = model.predict(frame)

            # Extract bounding box coordinates and confidence scores
            result = results[0]
            boxes = result.boxes.xyxy.tolist()
            confidences = result.boxes.conf.tolist()

            # Write bounding box coordinates to the output file
            for box, conf in zip(boxes, confidences):
                if conf > 0.5:
                    f.write(f"{frame_number} {' '.join(map(str, box))}\n")                            

            # Increment frame number
            frame_number += 1

    # Release video capture
    cap.release()
    print('---'*50)

if __name__ == "__main__":
    # Define input and output directories
    input_dir = Path.cwd() / 'VS13'
    output_dir = Path.cwd() / 'datasets'

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through video directories
    for video_dir in input_dir.iterdir():
        if not video_dir.is_dir():
            continue
        
        # Iterate through video files
        for video_file in video_dir.glob('*.MP4'):
            # Create output directory for each video
            output_subdir = output_dir / video_dir.name
            os.makedirs(output_subdir, exist_ok=True)
            
            # Define output file path
            output_path = output_subdir / f'{video_file.stem}.txt'

            # Read annotation file to get the last time the vehicle appears
            annotation_file = video_dir / f"{video_file.stem}.txt"
            with open(annotation_file, 'r') as f:
                _, time = f.readline().strip().split()
            
            # Process the video and save bounding box coordinates
            detect_and_save_bounding_boxes(video_file, output_path, time=float(time))
