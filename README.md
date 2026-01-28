# Flower Image Classification (CNN – PyTorch)

A multi-class image classification project using a custom Convolutional Neural Network (CNN) built from scratch in PyTorch to classify flowers into 14 different categories.

## Overview
This project focuses on fine-grained image classification, where visual differences between classes are subtle. The model is trained on real-world RGB images and evaluated on unseen validation data.

## Dataset
- 14 flower categories
- Images organized using `ImageFolder`
- Train / validation split
- Dataset sourced from Kaggle (not included in repository)

## Model Architecture
Input (3×224×224) →
- Conv2D → ReLU → MaxPool
- Conv2D → ReLU → MaxPool
- Conv2D → ReLU → MaxPool
- Fully Connected Layer → 14 classes

Loss Function:
- CrossEntropyLoss (Softmax applied internally)

Optimizer:
- Adam

## Results
- Training Accuracy: ~96%
- Validation Accuracy: ~60%

The gap between training and validation accuracy highlights challenges of fine-grained classification and limited dataset size.

## Features
- Custom CNN built from scratch
- Real-world image preprocessing (resize, normalize)
- GPU-accelerated training
- Single-image inference on unseen images
- Honest evaluation and overfitting analysis

## How to Run

```bash
pip install -r requirements.txt
python train.py
