# Siamese Face Recognition
This project trains a **Siamese Network** on a subset of the **Pins Face Recognition dataset** using **triplet loss** to learn facial similarity.

![Siamese_Overall_Architecture](https://github.com/user-attachments/assets/ab6f2ef5-8970-4e43-a0d1-3b2b925bb2b9)

## Table of Contents
- [How Triplet Loss Works](#how-triplet-loss-works)
- [Pins Face Recognition Dataset](#pins-face-recognition-dataset)
- [Augmentation Pipeline](#augmentation-pipeline)
- [Training](#training)
  - [Config](#config)
  - [Result](#result)
- [Inference Samples](#inference-samples)
  - [Sample 1 (Alex Turner)](#sample-1-alex-turner)
  - [Sample 2 (Enrique Iglesias)](#sample-2-enrique-iglesias)


## How Triplet Loss Works

The loss operates on triplets: **Anchor (A)**, **Positive (P)**, and **Negative (N)**. It ensures:

$$
\|f(A) - f(P)\|_2^2 + \alpha < \|f(A) - f(N)\|_2^2
$$


Where $f(x)$ is the embedding, $\|\cdot\|_2$ is the Euclidean distance, and $\alpha$ is a margin. The goal is to make similar images closer and dissimilar images farther apart, enabling effective facial similarity learning.


## Pins Face Recognition Dataset
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition)

The **Pins Face Recognition dataset** contains **17,534 cropped images** of **105 celebrities** collected from **Pinterest**, offering diverse poses and lighting for training face recognition models.  

For this project, we used a **subset of 30 individuals** to reduce computational complexity while ensuring effective training on embedded systems.


| **Full Dataset Distribution** | **Chosen 30 Persons from Dataset** |
|--------------------------------|-------------------------------------|
| ![Full Dataset](https://github.com/user-attachments/assets/c4310ccb-98a6-44c9-a168-980665493ed0) | ![Chosen Subset](https://github.com/user-attachments/assets/d2149c60-57d5-46c1-b924-4ce457c6e7a4) |

## Augmentation Pipeline
```python
augmented_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])

```
![image](https://github.com/user-attachments/assets/9aa0f228-2cf4-4c96-9e47-6eacc93d6a9e)



## Training  
### Config  

- **5-Fold Stratified K-Fold Cross-Validation**  
- **Batch Size:** `32`
- **Epochs:** `20`
- **Learning Rate:** `1e-4`
- **Optimizer:** `Adam`  

### Result  

This is the training metrics for **Fold 1**. Other folds showed **similar performance**.  

![image](https://github.com/user-attachments/assets/27c83972-1e57-49af-b53e-f21225127092)

## Inference Samples
### Sample 1 (Alex Turner)
![image](https://github.com/user-attachments/assets/b7ace51a-f159-45f7-8955-c077290e33fc)

### Sample 2 (Enrique Iglesias)
![image](https://github.com/user-attachments/assets/ce7a07cb-0c3b-4ec3-aa85-7d6f52f6aee2)

