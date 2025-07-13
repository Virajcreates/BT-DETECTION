# Brain Tumor Detection using Ensemble Deep Learning Models

## Introduction
Welcome to the Brain Tumor Detection project! This repository provides a deep learning solution for classifying brain MRI images into four categories: glioma, meningioma, pituitary tumor, and no tumor. By leveraging an ensemble of pre-trained models—EfficientNet-B0, ResNet18, and MobileNetV3—we achieve robust and accurate predictions. The project includes a user-friendly script for testing the model on single images or batches, complete with visualizations for better understanding of the results.

## Features
- Ensemble model for high-accuracy brain tumor classification.
- Interactive testing script for single images or batches.
- Visual feedback with image display and confusion matrix.
- Easy-to-use with minimal setup required.

## Requirements
- Python 3.x
- PyTorch
- Torchvision
- efficientnet_pytorch
- Pillow
- Matplotlib
- Seaborn
- NumPy
- Scikit-learn

**Note:** A GPU is recommended for faster inference, but the script can also run on a CPU.

## Installation
1. Clone this repository:
   ```
   git clone https://github.com/Virajcreates/BT-Detection.git
   cd BT-Detection
   ```
2. Install the required libraries:
   ```
   pip install torch torchvision efficientnet_pytorch Pillow matplotlib seaborn numpy scikit-learn
   ```
3. Ensure the pre-trained model files (`resnet18_best.pth`, `efficientnet_b0_best.pth`, `mobilenet_v3_best.pth`) are placed in the same directory as the script.

## Usage
Run the script using:
```
python Ensemble_Final.py
```
You will be prompted to:
1. Choose the mode: `'single'` for testing a single image or `'batch'` for testing a batch of images.
2. Provide the path to the image file (for single mode) or the test directory (for batch mode).

### Single Image Testing
- Enter `'single'` when prompted.
- Provide the path to an image file (e.g., `'path/to/image.jpg'`).
- The script will display the image with the predicted class and print the probabilities for each class.

### Batch Testing
- Enter `'batch'` when prompted.
- Provide the path to a directory containing subfolders for each class (e.g., `'path/to/test_dir'`).
- The script will output a classification report and a confusion matrix.

## Contributing
Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
