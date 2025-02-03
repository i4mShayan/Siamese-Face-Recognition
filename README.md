# Siamese Face Recognition
This project trains a **Siamese Network** on a subset of the **Pins Face Recognition dataset** using **triplet loss** to learn facial similarity.

![Siamese_Overall_Architecture](https://github.com/user-attachments/assets/ab6f2ef5-8970-4e43-a0d1-3b2b925bb2b9)

### How Triplet Loss Works

The loss operates on triplets: **Anchor (A)**, **Positive (P)**, and **Negative (N)**. It ensures:

$$
\|f(A) - f(P)\|_2^2 + \alpha < \|f(A) - f(N)\|_2^2
$$


Where $f(x)$ is the embedding, $\|\cdot\|_2$ is the Euclidean distance, and $\alpha$ is a margin. The goal is to make similar images closer and dissimilar images farther apart, enabling effective facial similarity learning.


## Pins Face Recognition Dataset
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition)

![image](https://github.com/user-attachments/assets/f45ec071-b0fa-444a-beb1-61fdc4d167a3)

The Pins Face Recognition dataset is a collection of facial images designed for training and evaluating face recognition models. It contains images of 105 individuals, captured in a variety of poses and lighting conditions, offering good diversity that is particularly useful for training embedded-based models.

For this project, we used a subset of the dataset, selecting only 30 individuals out of the 105 available for training. This subset was chosen to reduce computational complexity while leveraging the dataset's diversity to ensure effective model training on low resources.


| **Full Dataset Distribution** | **Chosen 30 Persons from Dataset** |
|--------------------------------|-------------------------------------|
| ![Full Dataset](https://github.com/user-attachments/assets/c4310ccb-98a6-44c9-a168-980665493ed0) | ![Chosen Subset](https://github.com/user-attachments/assets/d2149c60-57d5-46c1-b924-4ce457c6e7a4) |

## Augmentation Pipline
```python
augmented_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Randomly crop and resize
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip horizontally
    transforms.RandomRotation(degrees=45),  # Random rotation
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust color properties
    transforms.ToTensor(),  # Convert to tensor
])

```

## Results
![image](https://github.com/user-attachments/assets/27c83972-1e57-49af-b53e-f21225127092)
