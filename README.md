Brain Tumor MRI Segmentation Model
Overview
This project implements a deep learning–based Brain Tumor MRI Segmentation system using convolutional neural networks.
The model is designed to automatically segment tumor regions from MRI scans, assisting in medical image analysis and decision support.

This repository is intended as a research and demonstration project and is part of my AI/ML portfolio.

Key Features

MRI image preprocessing and normalization
Binary tumor segmentation
PyTorch-based training and evaluation pipeline
Visualization of predicted tumor masks vs ground truth
Model & Approach
Framework: PyTorch
Architecture: CNN-based segmentation network (U-Net style)
Loss Function: Dice / BCE-based loss
Evaluation Metrics:
Dice Score
IoU (Jaccard Index)
Accuracy
Dataset
Due to GitHub file size limitations, the dataset is not included in this repository.

Brain MRI Tumor Dataset (Kaggle / Medical MRI datasets)
You can download a compatible dataset from: 🔗 https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

After downloading, organize the dataset as: ├── disease/ └── normal/

Results
Below are sample segmentation outputs produced by the model:

Ground Truth vs Prediction
Segmentation Result

The model successfully highlights tumor regions with clear boundary separation.

How to Run
bash pip install -r requirements.txt python testing.py
