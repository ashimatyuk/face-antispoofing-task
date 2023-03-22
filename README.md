# face_antispoofing

This repository contains implementation of the EfficientNet_B0 model for spoofing classification task.

# Dataset

Dataset consists of 5000 train images and 968 validtion images.

# Model

EfficientNet_B0 model, [best.model](https://drive.google.com/file/d/1SfjYAKnuFtGZFrcFIT3UlmFDpWq21pZY/view?usp=share_link)

# Augmentations

GaussNoise, GaussianBlur, Colorjitter, FancyPCA- one of all with p=0.75. RandomRotate90 p=0.3, ShiftScaleRotate p=0.7. The goal was to avoid overfitting, that's why I set augs up to let them appear more frequently than default p=0.5.

# Data preprocessing

Train data was imbalanced- around 30/70 towards spoof class. Firstly, I calculated sampleweights and used them with sampler in Dataloader, but it didn't affect much and then I applied undersampling to reach 50% live/50 spoof.

# Evaluation

The problem is that val_dataset has a huge imbalance: only 68 samples are spoof and 900 are live. After some experiments with valid_recall near 0.69 I decided to reduce its threshold. Below are logs.

Undersampled data (50/50), dropout(0.3), recall(0.3), prec(0.6)  [clear.ml ](https://app.clear.ml/projects/54c8c155bd634934a727296968816835/experiments/260881bdf8454102a8bb4de9896137d8/output/execution) 

Undersampled data (40 live/60 spoof), dropout(0.3), recall(0.3), prec(0.6)  [clear.ml ](https://app.clear.ml/projects/54c8c155bd634934a727296968816835/experiments/3c324a05e06d465fb1f17e1c34a9f9cf/output/execution)

At second one I decided to add to a train data a bit more spoof samples to let model find them better in imbalanced validation dataset. Valid_metrics became better: from loss 0.076, precision 0.77, recall 0.75 to loss 0.0642 precision 0.85 recall 0.76

P.S. I also trained model on cropped faces with given bboxes in csv and and returned bboxes from retinaface model- and there was no improvement.

# Test_images

Folder test_images contains personal made photos I used to pass through the model. Here are the results (if image name contains 'spoof', label should be 1):

![image](https://user-images.githubusercontent.com/102593339/226798873-a1806ef4-1d40-45a0-85fd-b61a9fe9fb4b.png)

# Repository_structure

main.py - training process with Catalyst 
custum_metrics.py - custom metrics for Catalyst BatchMetricCallback
balancing_dataset.py - balance data to (40 live/60 spoof)
model.py - EfficientNet_B0 model class
dataset.py - custom dataset class
const.py - config with constants
test_predictions.py - function to predict labels from test_images folder
retina_crop.py - saves scaled cropped faces
dummy_functions.py - some functions used during experiments

# Instruction
To repeat best_model experiment do the following:
1. Download dataset and place it in the main folder like main_folder/celeba/train, test, train.csv, test.csv
2. Run balancing_dataset.py
3. Run main.py (clear ml task will be created, see 31-32 rows)
4. Place model.best.pth in the main folder (optionaly)
4. Run test_predictions.py (optionaly)
