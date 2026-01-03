# Brain Tumor MRI Image Classification Using a OvF Strategy with CNNs

## Overview

Brain tumors are abnormal growths of cells within or surrounding the brain and can be either benign or malignant. Magnetic Resonance Imaging (MRI) is one of the most commonly used imaging modalities for brain tumor diagnosis due to its high soft-tissue contrast and detailed anatomical information.

Brain tumor diagnosis from MRI scans is a multi-class classification problem that plays a critical role in clinical decision-making. Conventional deep learning approaches typically address this task using standard multi-class classification models. However, in medical diagnosis, models are usually evaluated through Receiver Operating Characteristic (ROC) Curve Analysis and Decision Curve Analysis (DCA), and each model's predictions are typically based on a decision threshold. These evaluation methods and decision rules are inherently defined for binary classification problems. Therefore, instead of using one single multi-class classification model, this study will use the **One-vs-Followers (OvF) approach** to decompose the multi-class problem into a sequence of binary classification tasks in order to enable the integration of ROC Curve Analysis, DCA, and optimal classification thresholds.

This project implements a OvF approach combined with Convolutional Neural Networks (CNNs) to automatically classify brain tumor status using axial slices from brain MRI images as part of my master's thesis. The task is formulated as a multi-class image classification problem, distinguishing between 4 brain tumor categories: *No tumor, Glioma, Meningioma,* and *Pituitary tumor*. This study aims to develop a deep learning–based approach that can assist in brain tumor scanning by improving diagnostic efficiency and consistency. The code was developed and executed using RStudio.

## Data

This project uses the Brain Tumor MRI dataset, which contains 7023 brain MRI images across multiple anatomical slice orientations (*axial, coronal, sagittal*), collected from individuals undergoing diagnostic evaluation. The images are divided into 4 classes: *NoTumor, Glioma, Meningioma,* and *Pituitary*. The dataset is publicly available at: <https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset>

For model training and evaluation, only the axial slices were used, comprising a total of 3,569 images.

(List data preprocessing steps (normalization, resizing, data split ...))

## Methods

(Explain ROC, AUC, 3 criteria, DCA, optimal threshold, OvR (and why it can't work), OvF, ... with citations)

## Results

## References
