# Assignment 2 – CNN Interpretability

## Overview

This assignment explores how a pretrained Convolutional Neural Network (CNN) makes predictions using Class Activation Maps (CAM). The goal is to understand not only what the model predicts, but also where it focuses in the image.

## Model

ResNet18 pretrained on the ImageNet dataset.

## Goal

- Generate predictions for different images  
- Visualize model attention using CAM  
- Understand which parts of the image influence the prediction  
- Compare correct and incorrect predictions  
- Analyze model behavior on unseen classes 

## Approach

For each image:
- The model predicts a class  
- A CAM heatmap is generated  
- The heatmap is overlaid on the original image  
- Different layers (layer1–layer4) are analyzed to understand how attention evolves  

## Data

All images used in this assignment were collected from Pexels.

## Results

The results include:
- Positive examples (correct prediction with good focus)  
- Weak positive examples (correct prediction with partial focus)  
- Negative examples (incorrect prediction)  
- Negative examples (incorrect prediction with good focus) 
- Out-of-distribution example (LEGO)
