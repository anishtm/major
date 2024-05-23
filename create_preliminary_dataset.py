'''
This program is to get the annotations to fine tune the YOLOv5 model to accurately and consistently detect cars in VS13.'''
import cv2
from ultralytics import YOLO
from pathlib import Path
import os

# Load the YOLOv5 model
model = YOLO('yolov5m.pt')

def detect_and_save_bounding_boxes(video_path, output_path=None, time=None):
    """
    Detects cars in a video and saves bounding boxes to a file.

    Parameters:
    - video_path (str): Path to the video file.
    - output_path (str): Path to the output text file. Defaults to video filename with .txt extension.
    - time (float): Time in seconds to stop processing the video. If None, processes the entire video.
    """
    frame_number = 0
    
    if not output_path:
        output_path = f'{Path(video_path).stem}.txt'
        
    cap = cv2.VideoCapture(str(video_path))
    if time:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_to_stop = int(time * fps) + 1
    else:
        frame_to_stop = None
    
    with open(output_path, 'w') as f:
        while True:
            ret, frame = cap.read()
            if not ret or (frame_to_stop is not None and frame_number >= frame_to_stop):
                break
            
            height, width, _ = frame.shape
            
            # Perform object detection on the frame
            results = model.predict(frame)
            result = results[0]
            boxes = result.boxes.xyxy.tolist()
            classes = result.boxes.cls.tolist()
            confidences = result.boxes.conf.tolist()

            # Iterate through the results
            for box, cls, conf in zip(boxes, classes, confidences):
                if conf > 0.5 and cls == 2:  # Class 2 typically represents 'car' in COCO dataset
                    f.write(f"{frame_number} {' '.join(map(str, box))}\n")
            
            # Increment frame number for the next iteration
            frame_number += 1

        cap.release()

def process_videos_in_directory(input_dir, output_dir):
    """
    Processes all videos in the input directory and saves bounding box data to the output directory.

    Parameters:
    - input_dir (Path): Path to the directory containing the video files.
    - output_dir (Path): Path to the directory to save the output text files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for subdir in input_dir.glob('*'):
        if subdir.is_dir():
            for vid in subdir.glob('*.MP4'):
                video_output_dir = output_dir / subdir.stem
                os.makedirs(video_output_dir, exist_ok=True)
                output_path = video_output_dir / f'{vid.stem}.txt'
                print(f'Processing {vid.stem}')
                
                txt_file = subdir / f"{vid.stem}.txt"
                with open(txt_file, 'r') as f:
                    line = f.readline().strip()
                    _, time = line.split()
                
                detect_and_save_bounding_boxes(vid, output_path=output_path, time=float(time))

if __name__ == "__main__":
    cur_dir = Path.cwd()
    input_dir = cur_dir / 'VS13'
    output_dir = cur_dir / 'preliminary_dataset'
    
    process_videos_in_directory(input_dir, output_dir)
