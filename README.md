# Food101 Subset Sampling & Classification

This repository contains a project for food image classification using a subset of the Food101 dataset.

## Dataset Overview

The **Food101** dataset comprises:
- **101 food categories**
- **101,000 images in total**
  - **750 training images** per class
  - **250 manually reviewed test images** per class

This project specifically works with a **balanced random 10% sample** of the Food101 dataset, ensuring that each food category is equally represented in both the training and testing subsets.

## Project Overview

This notebook/project demonstrates:
- How to download and use the Food101 dataset.
- How to randomly sample 10% of the training and test data, maintaining category balance.
- Basic exploratory data analysis and visualization of the sampled data.
- Implementation of a classification model using PyTorch and related libraries.

## Setup Instructions

### 1. Clone the repository:
```bash
git clone <repository-url>
cd Foodie-Vision
```

### 2. Create and activate a virtual environment:

#### Using Conda (custom location):
```bash
conda create --prefix ./venv python=3.12.9
conda activate ./venv
```

#### Or using Python's venv:
```bash
python -m venv venv
source venv/Scripts/activate  # For Git Bash/Unix
```

### 3. Install required packages:

The dependencies are listed in `requirements.txt`. Install them by running:
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

This command uses the extra PyTorch index to ensure the proper CUDA-enabled versions of `torch`, `torchvision`, and `torchaudio` are installed along with other standard packages.

<!-- ## Requirements

The `requirements.txt` file includes:
- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn
- mlxtend
- pillow
- torch
- torchvision -->

<!-- ## Usage

### 1. Data Preparation
The notebook includes code to download the Food101 dataset (if not already available) and to create balanced training and test splits by randomly sampling 10% of the images per category.

### 2. Model Training & Evaluation
The project demonstrates a simple image classification pipeline using PyTorch, including data loading, model training, and evaluation.

### 3. Exploratory Data Analysis
Visualizations and analysis of the sampled data are provided to better understand the distribution of food categories and sample images.

## Additional Information

- **Balanced Sampling:**  
  The sampling process ensures that each of the 101 food categories is represented equally in the training and test splits.
- **Dataset Subset:**  
  Working with a 10% subset significantly reduces the computational requirements while still providing a diverse set of samples across all food categories. -->


