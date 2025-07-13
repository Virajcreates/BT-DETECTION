import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from torch.amp import autocast
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import classification_report, confusion_matrix

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 4
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['glioma', 'meningioma', 'pituitary', 'no_tumor']

# Define individual models
class BaseModel(nn.Module):
    def __init__(self, arch, pretrained=True, num_classes=NUM_CLASSES):
        super(BaseModel, self).__init__()
        self.arch = arch

        if arch == 'efficientnet_b0':
            self.model = EfficientNet.from_pretrained('efficientnet-b0')
            num_features = self.model._fc.in_features
            self.model._fc = nn.Linear(num_features, num_classes)
        elif arch == 'resnet18':
            self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
        elif arch == 'mobilenet_v3':
            self.model = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            num_features = self.model.classifier[0].in_features
            self.model.classifier = nn.Sequential(nn.Linear(num_features, num_classes))
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        # Freeze all base model layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze classifier head
        if arch == 'resnet18':
            for param in self.model.fc.parameters():
                param.requires_grad = True
        elif arch == 'efficientnet_b0':
            for param in self.model._fc.parameters():
                param.requires_grad = True
        elif arch == 'mobilenet_v3':
            for param in self.model.classifier[0].parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)

# Ensemble Model that loads pre-trained individual models
class EnsembleModel(nn.Module):
    def __init__(self, resnet_path, efficientnet_path, mobilenet_path, num_classes=NUM_CLASSES):
        super(EnsembleModel, self).__init__()

        self.resnet = BaseModel('resnet18').to(DEVICE)
        self.resnet.load_state_dict(torch.load(resnet_path, map_location=DEVICE))

        self.efficient = BaseModel('efficientnet_b0').to(DEVICE)
        self.efficient.load_state_dict(torch.load(efficientnet_path, map_location=DEVICE))

        self.mobilenet = BaseModel('mobilenet_v3').to(DEVICE)
        self.mobilenet.load_state_dict(torch.load(mobilenet_path, map_location=DEVICE))

        # Freeze all models
        self.models = nn.ModuleList([self.resnet, self.efficient, self.mobilenet])
        for model in self.models:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, x):
        outputs = []
        for model in self.models:
            with autocast('cuda'):
                output = model(x)
                outputs.append(torch.softmax(output, dim=1))
        avg_output = torch.stack(outputs).mean(dim=0)
        return avg_output

# Function to test a single image
def test_single_image(model, image_path, class_names):
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = output[0]
        predicted_class = torch.argmax(probabilities).item()

    # Denormalize for visualization
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    image_vis = inv_normalize(image_tensor[0]).cpu()
    image_vis = transforms.ToPILImage()(image_vis)

    # Plot the image with prediction
    plt.imshow(image_vis)
    plt.title(f"Predicted: {class_names[predicted_class]}")
    plt.axis('off')
    plt.show()

    # Print probabilities
    print("Probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"{class_names[i]}: {prob:.4f}")

# Function to test a batch of images
def test_batch(model, test_dir, class_names):
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute and print classification report
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Main function with interactive prompts
def main():
    # Prompt for mode
    while True:
        mode = input("Enter mode (single/batch): ").lower()
        if mode in ['single', 'batch']:
            break
        print("Invalid mode. Please enter 'single' or 'batch'.")

    # Prompt for input path
    while True:
        input_path = input(f"Enter the path to the image (for single mode) or test directory (for batch mode): ").strip()
        if os.path.exists(input_path):
            break
        print(f"Path does not exist: {input_path}. Please enter a valid path.")

    # Model paths (assuming they are in the current directory)
    resnet_path = r"C:\Users\viraj\Documents\Virajs Projects\BT DETECTION\models\resnet18_best.pth"
    efficientnet_path = r"C:\Users\viraj\Documents\Virajs Projects\BT DETECTION\models\efficientnet_b0_best.pth"
    mobilenet_path = r"C:\Users\viraj\Documents\Virajs Projects\BT DETECTION\models\mobilenet_v3_best.pth"


    # Check if model files exist
    for path in [resnet_path, efficientnet_path, mobilenet_path]:
        if not os.path.isfile(path):
            print(f"Model file not found: {path}. Please ensure all pre-trained models are in the current directory.")
            return

    # Create ensemble model
    model = EnsembleModel(resnet_path, efficientnet_path, mobilenet_path)
    model.to(DEVICE)

    if mode == 'single':
        if not input_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            print("Invalid file format. Please provide a valid image file (e.g., .jpg, .jpeg, .png).")
            return
        test_single_image(model, input_path, CLASS_NAMES)
    elif mode == 'batch':
        if not os.path.isdir(input_path):
            print("Invalid directory. Please provide a valid test directory with subfolders for each class.")
            return
        test_batch(model, input_path, CLASS_NAMES)

if __name__ == "__main__":
    main()