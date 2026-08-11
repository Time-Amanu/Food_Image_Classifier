# 🍕 Food-101 Image Classifier with Transfer Learning

*A ResNet18-based image classifier that identifies 101 categories of food, built with PyTorch using transfer learning.*

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Results and Visualizations](#results-and-visualizations)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Future Improvements](#future-improvements)
- [License](#license)

## Overview

This project fine-tunes a pretrained **ResNet18** to classify food images across the 101 categories in the [Food-101 dataset](#dataset). The convolutional base is frozen and only a new final layer is trained — a fast, lightweight transfer-learning approach that works well even with limited compute, helped along by training on a 10% stratified subset of the data (see [Dataset](#dataset)).

The code is organized as a modular pipeline — separate scripts for data loading, model building, training, and utilities — alongside a notebook that walks through the full workflow end-to-end, from downloading the data to training, evaluating, and generating predictions.

## Project Structure

```
food-101-classifier/
├── data_setup.py              # Custom Dataset class + data-loading utilities
├── model_builder.py           # Builds the ResNet18 transfer-learning model
├── engine.py                  # train_step() / test_step() training & evaluation loops
├── utils.py                   # Plotting utilities (loss & accuracy curves)
├── main.py                    # Pipeline entry point
├── requirements.txt           # Python dependencies
└── Food101_Classifier.ipynb   # Full notebook: EDA, training, evaluation, predictions
```

| File | Responsibility |
|---|---|
| `data_setup.py` | Defines `FoodDataset`, a custom PyTorch `Dataset` for loading Food-101 images and labels, plus a helper for parsing the dataset's train/test split files. |
| `model_builder.py` | `create_model()` loads a pretrained ResNet18, freezes its base layers, and replaces the final layer with one sized for the target number of classes (101, in this project). |
| `engine.py` | `train_step()` and `test_step()` — modular, single-epoch training and evaluation loops that return loss and accuracy. |
| `utils.py` | `plot_loss_curves()` for visualizing training/validation loss and accuracy over epochs. |
| `main.py` | Imports the modular components, sets hyperparameters, and serves as the pipeline's entry point. |

> Rename `Food101_Classifier.ipynb` above to match your actual notebook filename.

## Dataset

This project uses the **[Food-101 dataset](https://www.kaggle.com/datasets/dansbecker/food-101)**, downloaded via `kagglehub`. Originally introduced by Bossard et al. (2014), it contains:

- **101** food categories
- **101,000** total images (750 training + 250 test images per class)

To keep training fast and efficient, this project trains on a **10% stratified subset** of the training data — sampled to preserve the same per-class balance as the full dataset — rather than the entire training set.

## Results and Visualizations

**Training Configuration**

| Hyperparameter | Value |
|---|---|
| Epochs | 4 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Device | CUDA if available, else CPU |

**Performance**

> Add your own numbers once training is complete.

| Metric | Value |
|---|---|
| Best Validation Accuracy | `[e.g., 72.5%]` |
| Best Epoch | `[e.g., 4]` |

**Training & Validation Curves**  
*(Generated with `utils.plot_loss_curves()`)*

![Loss and accuracy curves](assets/loss_accuracy_curves.png)

**Sample Prediction**  
*(Generated with `predict_and_show()`)*

![Sample prediction](assets/sample_prediction.png)

```python
# Run inference on your own image
# (predict_and_show is currently defined in the notebook —
#  see Future Improvements for turning it into a standalone predict.py)
import torch
from model_builder import create_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open("classes.txt") as f:
    classes = [line.strip() for line in f]

model = create_model(num_classes=101)
model.load_state_dict(torch.load("best_food_classifier.pth", map_location=DEVICE))
model.to(DEVICE)

predict_and_show("path/to/your/image.jpg", model, classes, DEVICE)
```

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/food-101-classifier.git
cd food-101-classifier
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```
> Requires Python 3.9+. For GPU support, install the PyTorch build matching your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/) before running the command above.

**3. Download the dataset**
```python
import kagglehub
path = kagglehub.dataset_download("dansbecker/food-101")
```

**4. Run the pipeline**
```bash
python main.py
```
This wires up the modular components and confirms the pipeline is ready to go. The full training loop — including metric tracking and checkpoint saving to `best_food_classifier.pth` — is demonstrated end-to-end in the notebook.

## Tech Stack

- 🔥 **PyTorch** — model definition & training
- 🖼️ **Torchvision** — pretrained ResNet18 and image transforms
- 🔢 **NumPy** — numerical operations
- 📊 **Matplotlib** — training curve visualizations
- 🖌️ **Pillow (PIL)** — image loading
- 📦 **kagglehub** — dataset download
- ⏳ **tqdm** — progress bars
