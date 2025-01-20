# Technical Report for Deep Learning-based Pancreas Cancer Segmentation and Classification with nnUNetv2 Multi Task Learning
## Introduction

This report describes the development of a multi-task deep learning model for pancreas cancer segmentation and classification using 3D CT scans. The model leverages the nnUNetv2 framework, known for its state-of-the-art performance in biomedical image segmentation, with modifications to include a classification head. The primary objective is to achieve accurate segmentation of the pancreas and lesions, along with lesion subtype classification into three categories.

## Environments and Requirements

For detailed environment and requirements, refer to the official nnU-Net documentation: Installation Instructions.
After initial nnUNetv2 framework installation, you can download the folders: Files to add_nnunetv2, Files to replace_nnunetv2 and File_to_replace_sitepackages/dynamic_network_architecture

### Setting up the environment
After downloading those folders, you need to add or replace the content of the folder to the correct directory. Where to add/replace the files is match the same directory to how nnUNetv2 framework is structured.
For instance,
The file for data_loader_3d.py is in "File to replace_nnunetv2" in this directory: ``` File to replace_nnunetv2/training/dataloading ```
Which should replace the same filename that is in the nnUNetv2 framework: ``` nnunetv2/training/dataloading ```

For the file in File_to_replace_sitepackages/dynamic_network_architecture (unet.py), that file should replace in the directory of your environment
``` envs/<environment which you run nnUNetv2>/Lib/site-packages/dynamic_network_architecture/architecture ```

## Dataset:
Data preparation: the Dataset was copied into the nnUNet_raw folder following this structure:
``` nnUNet/ ├──nnUNet_results | ├──nnUNet_preprocessed | ├──nnUNet_raw || ├──Dataset001_Pancreas ||| ├──imagesTr (images for training and validation data) ||| ├──imagesTs (images for test data) ||| ├──labelsTr  (masks for training and validation data)

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
Use the 
