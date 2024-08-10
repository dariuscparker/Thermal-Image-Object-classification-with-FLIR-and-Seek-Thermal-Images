# **Thermal Image Object Classification with FLIR and Seek Thermal Images**

This repository contains the code and resources for an image classification task using thermal images captured by FLIR and Seek Thermal cameras. The goal is to classify images into three categories: **cat, car,** and **man** using a Vision Transformer (ViT) model.

## **Dataset**

The dataset used in this project comprises images from two different thermal cameras:
- **Seek Thermal**: 6414 images with a resolution of 300 x 400.
- **FLIR**: 1014 images with a resolution of 1080 x 1440.

The dataset can be downloaded from the following link:  

[**Thermal Image dataset for object classification**](https://data.mendeley.com/datasets/btmrycjpbj/1)

### **Dataset Structure**
The dataset is organized into training and test sets for each camera, with subdirectories for each class (`cat`, `car`, `man`). The data sets are combined for training.

## **Project Overview**

This project demonstrates the following steps:
1. **Data Preparation**: 
   - Extracting the dataset from a zip file stored on Google Drive.
   - Renaming directories for consistency.
   - Visualizing sample images from each class.
   - Checking the distribution of classes to ensure a balanced dataset.
   - Merging datasets from both cameras into a single training set.

2. **Model Training**:
   - Using a pre-trained Vision Transformer (ViT) model from Hugging Face.
   - Modifying the model's classifier to suit the task.
   - Training the model using PyTorch with early stopping to prevent overfitting.
   - Evaluating the model's performance on a validation set.

3. **Visualization and Evaluation**:
   - Displaying sample predictions to verify model accuracy.
   - Plotting label distribution to ensure balance.

## **Getting Started**

### **Prerequisites**
- Python 3.x
- [Google Colab](https://colab.research.google.com/) (Recommended for GPU access)
- Required Python packages (can be installed using `pip`):
    ```bash
    pip install torch torchvision transformers matplotlib tqdm
    ```

### **Running the Notebook**

It is recommended to run this notebook in a Google Colab environment, with the use of a GPU. You can view and run the notebook directly in Google Colab by clicking the link below:

[**Open in Google Colab**](https://colab.research.google.com/drive/1qTHWKclgj_ENl8ccEclF0swiXiQRAM9i#scrollTo=B5u4ad2LHzbP)


### **Dataset Access**
Since the dataset is large, it is not included in this repository. You can download it from the [Mendeley Data link](https://data.mendeley.com/datasets/btmrycjpbj/1) This link can also be found in the notebook.

## **Results**

The Vision Transformer model achieved a validation accuracy of **99.5%**, demonstrating its effectiveness for this task. Sample predictions are provided in the notebook to verify the model's performance visually.

## **Acknowledgements**

- **Hugging Face** for providing the pre-trained Vision Transformer model.
- **FLIR Systems** and **Seek Thermal** for their contributions to the dataset.

## **Contributing**

If you have any suggestions or improvements, feel free to submit a pull request or open an issue. All contributions are welcome!

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
