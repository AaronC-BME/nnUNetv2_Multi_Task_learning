# Technical Report for Deep Learning-based Pancreas Cancer Segmentation and Classification with nnUNetv2 Multi Task Learning
## Introduction

This report describes the development of a multi-task deep learning model for pancreas cancer segmentation and classification using 3D CT scans. The model leverages the nnUNetv2 framework, known for its state-of-the-art performance in biomedical image segmentation, with modifications to include a classification head. The primary objective is to achieve accurate segmentation of the pancreas and lesions, along with lesion subtype classification into three categories.

## Environments and Requirements

For detailed environment and requirements, refer to the official nnU-Net documentation: [Installation Instructions](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md).
After initial nnUNetv2 framework installation, you can download the folders: Files to add_nnunetv2, Files to replace_nnunetv2 and File_to_replace_sitepackages/dynamic_network_architecture

### Setting up the environment
After downloading those folders, you need to add or replace the content of the folders to the correct directory. Where to add/replace the files matches the same directory to how nnUNetv2 framework is structured.
For instance,
The file for data_loader_3d.py is in "File to replace_nnunetv2" in this directory: ``` File to replace_nnunetv2/training/dataloading ```
Which should replace the same filename that is in the nnUNetv2 framework: ``` .../nnunetv2/training/dataloading ```

For the file in File_to_replace_sitepackages/dynamic_network_architecture (unet.py), that file should replace in the directory of your environment
``` envs/<environment which you run nnUNetv2>/Lib/site-packages/dynamic_network_architecture/architecture ```

## Dataset:
Data preparation: the Dataset was copied into the nnUNet_raw folder following this structure:
```
nnUNet/ 
├──nnUNet_results 
├──nnUNet_preprocessed 
├──nnUNet_raw 
    ├──Dataset001_Pancreas 
        ├──imagesTr (images for training and validation data) 
        ├──imagesTs (images for test data) 
        ├──labelsTr  (masks for training and validation data)
```

Create an excel file of the data in /imagesTr in this format:  filename    subtype    split
Or use the classification_label.csv file in this repository (inside the Data_prep folder)

Use the notebook ```Creating_metadata.ipynb``` to create the metadata needed for preprocesssing

## Preprocessing
Preprocessing was completed by using the nnUNetv2_plan_and_preprocess command
Preprocessing steps include:
•	CT Normalization using foreground intensity statistics.
•	Resampling to a standard spacing of [2.0, 0.73, 0.73].
•	Transposing axes to align with nnUNet conventions.
•	Cropping non-zero regions for efficient computation.

Run the command in the nnUNetv2 environment for data preprocessing:
``` nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity ```

## Training
Before training start, the split_final.json need to be modified to explicitly split training and validation data
Use the ``` creating_splits.ipynb ``` to create explicit splits, this notebook should create a new split_final.json that replace the existing split_final.json generated in the ``` nnUNet_preprocessed ``` folder

Run the following commnad to start training the model:
``` nnUNetv2_train 1 3d_fullres 0 -tr nnUNetMultiTrainer_v2 ```

Fold 0 was used to ensure that the validation set is only used for validation once the training is complete

If for some reason the training was completed but it did not run the validation, use the following command:
``` nnUNetv2_train 1 3d_fullres 0 -tr nnUNetMultiTrainer_v2 --val ```

## Inference
Before the start of inference, the plans.json file need to be modified, since we are only running 3d_fullres, change the ``` network_class_name ``` from "dynamic_network_architectures.architectures.unet.PlainConvUNet" to "dynamic_network_architectures.architectures.unet.PlainConvUNetWithClassification" under 3d_fullres/architecture

Use the following notebook to filter out all the unnessary state_dict keys and save a new checkpoint: ``` Filter_state_dict.ipynb ```

Run the following command for inferece (for -chk, use path to the checkpoint_final_filtered.pth can also work)
``` nnUNetv2_predict -i <INPUT PATH TO TEST SET> -o <OUTPUT PATH TO SAVE THE RESULT> -d 1 -c 3d_fullres -f 0 -chk checkpoint_final_filtered.pth ```

## Evaluation
Evaluation is done at the end of each epoch during training and at the end of validation.
Evaluation Metrics during training:
  - Segmentation
    - Dice score coefficient per class
    - Hausdorff distance
    - Volumetric difference
    - training loss
    - validation loss
  - Classification
    - Accuracy
    - Precision
    - Recall
    - F1_score
    - Macro-averaged_F1_score
    - classification loss
Evaluation Metric for validation:
  - Segmentation:
    - Mean dice score
    - Dice score coefficient per class
  - Classification
    - Accuracy
    - Precision
    - Recall
    - F1_score
    - Macro-averaged_F1_score
    - Confusion Matrix
   
## Results
Validation Results:
  - Segmentation:
    -	Mean Dice Score (Pancreas+Lesion): 0.7526710958310098
    -	Dice Score (Pancreas): 0.8937863042679479
    -	Dice Score (Lesion): 0.6115558873940716
  - Classification:
    -	Accuracy: 0.75
    -	Precision: 0.7652116402116403
    - Recall: 0.75
    -	F1_score: 0.7430555555555556
    -	Macro-averaged_F1_score: 0.7388888888888889
    -	Confusion Matrix: [ 6,  2,  1],
                        [ 0, 14,  1],
                        [ 1,  4,  7]

  
