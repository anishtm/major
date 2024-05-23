# Vehicle Speed Estimation with YOLO Detector and RNN Estimator

## Project Overview

This project aims to estimate vehicle speeds using a combination of the YOLOv5 object detector and a Recurrent Neural Network (RNN) estimator. The process involves detecting cars in video frames, extracting bounding box information, and then using an RNN to predict vehicle speeds based on the changes in bounding box areas over time.

## Steps and Methodology

### Step 1: Acquire Dataset "VS13"
- The dataset "VS13" was obtained, which contains videos of various vehicles.

### Step 2: Detect Cars with YOLOv5
- YOLOv5 from Ultralytics was used to detect cars in the VS13 dataset.

### Step 3: Fine-Tune YOLOv5
- YOLOv5 was fine-tuned with the images from VS13, using the annotations generated from the initial detection, to improve detection accuracy specific to this dataset.

### Step 4: Detect Cars and Record Bounding Boxes
- Cars were detected in each frame of the videos, and the bounding box coordinates were recorded for each detected car.

### Step 5: Preprocess Data
- The following preprocessing steps were performed on the data:
  - Removal of non-consecutive frames
  - Removal of null values
  - Calculation of bounding box areas
  - Calculation of changes in bounding box areas between consecutive frames

### Step 6: Create and Train RNN Model
- A new RNN model was created and trained using the processed data.

### Step 7: Test the Model
- The model was tested, and its performance was evaluated based on the Root Mean Square Error (RMSE).

## Training Performance

During training, a significant reduction in RMSE was observed:
- Initially, the RMSE was in the 60s.
- The RMSE rapidly decreased, reaching approximately 10 within the first 300 epochs.
- After 600 epochs, the RMSE stabilized at around 4, indicating effective learning by the model.

The training curve is illustrated in Figure 1.

## Testing Performance

On the test set, the model achieved an RMSE of 8.1010. This demonstrates that the model generalizes well to unseen data, maintaining a relatively low error rate in speed estimation.

## Per Vehicle Performance

The model's performance was further analyzed across different vehicle types within the dataset, consisting of 13 vehicles and 400 videos. The RMSE values for individual vehicles are listed in the table below:

| Car                | RMSE   |
|--------------------|--------|
| Peugeot307         | 5.5319 |
| RenaultCaptur      | 6.7812 |
| Peugeot208         | 4.3728 |
| NissanQashqai      | 5.3187 |
| MercedesAMG550     | 4.0602 |
| MercedesGLA        | 7.4322 |
| CitroenC4Picasso   | 4.4368 |
| KiaSportage        | 5.4585 |
| RenaultScenic      | 7.6198 |
| Peugeot3008        | 6.2481 |
| OpelInsignia       | 6.0126 |
| Mazda3             | 6.4512 |
| VWPassat           | 5.5866 |
| **Average**        | 5.7931 |

## Conclusion

The combination of YOLOv5 for object detection and an RNN for speed estimation has proven effective in this project. The model achieved a training RMSE of approximately 4 and a testing RMSE of 8.1010. The analysis of per vehicle performance shows that the model performs well across different vehicle types, with an average RMSE of 5.7931.

## Acknowledgments

We would like to thank Ultralytics for providing YOLOv5 and the creators of the VS13 dataset for making this project possible.

---

For any questions or further information, please contact the project contributors.

**Contributors:**
- [Your Name]
- [Collaborator's Name]
- [Additional Contributor's Name]

**Date:**
- May 23, 2024

**License:**
- [Specify License Here]

**References:**
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [VS13 Dataset Source](Provide Source Link)