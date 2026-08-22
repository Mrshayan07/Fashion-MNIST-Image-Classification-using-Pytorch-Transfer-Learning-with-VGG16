# Fashion-MNIST Image Classification using VGG16

This project implements an image classification model using **Transfer Learning** with a pretrained **VGG16** model in PyTorch.

The pretrained VGG16 convolutional layers are frozen, and the classifier is replaced and trained to classify images from the **Fashion-MNIST dataset** into 10 different categories.

## 🚀 Project Overview

The workflow includes:

* Loading and exploring the Fashion-MNIST dataset
* Visualizing sample images
* Splitting the dataset into training and testing sets
* Creating a custom PyTorch `Dataset`
* Applying image transformations
* Converting grayscale images to RGB
* Using a pretrained VGG16 model
* Freezing pretrained feature extraction layers
* Replacing the classifier for 10-class classification
* Training the model using Cross-Entropy Loss and Adam optimizer
* Evaluating model performance on training and test data

## 🧠 Model Architecture

The project uses a pretrained **VGG16** model.

### Transfer Learning Strategy

* Pretrained VGG16 feature extraction layers are frozen.
* Fashion-MNIST grayscale images are converted to 3-channel RGB images.
* Images are resized to `224 × 224`.
* The original VGG16 classifier is replaced with a custom classifier.

```text
Input Image
     ↓
Resize to 224 × 224
     ↓
Pretrained VGG16 Feature Extractor
(Frozen)
     ↓
Linear Layer (25088 → 1024)
     ↓
ReLU + Dropout
     ↓
Linear Layer (1024 → 512)
     ↓
ReLU + Dropout
     ↓
Linear Layer (512 → 10)
     ↓
Predicted Fashion Category
```

## 🛠️ Technologies Used

* Python
* PyTorch
* Torchvision
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* PIL

## ⚙️ Training Configuration

| Parameter         | Value             |
| ----------------- | ----------------- |
| Model             | VGG16             |
| Approach          | Transfer Learning |
| Optimizer         | Adam              |
| Learning Rate     | 0.0001            |
| Loss Function     | CrossEntropyLoss  |
| Batch Size        | 32                |
| Epochs            | 10                |
| Number of Classes | 10                |

## 📂 Dataset

The project uses a subset of the **Fashion-MNIST dataset** stored in CSV format.

* The first column represents the class label.
* The remaining 784 columns represent pixel values of a `28 × 28` grayscale image.

## 🔄 Data Preprocessing

The following preprocessing steps are applied:

1. Reshape flattened pixel values into `28 × 28` images.
2. Convert image data to `uint8`.
3. Convert grayscale images to 3-channel RGB.
4. Convert the NumPy array to a PIL image.
5. Resize images to `256`.
6. Center crop images to `224 × 224`.
7. Convert images to PyTorch tensors.
8. Normalize images using ImageNet normalization values.

## 🏋️ Training

The model is trained using:

```python
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    vgg16.classifier.parameters(),
    lr=0.0001
)
```

Only the custom classifier layers are trained while the pretrained VGG16 feature extraction layers remain frozen.

## 📊 Evaluation

After training, the model is evaluated on:

* Test Dataset
* Training Dataset

Accuracy is calculated using:

```text
Accuracy = Correct Predictions / Total Predictions
```

## 📁 Project Structure

```text
Fashion-MNIST-VGG16-Transfer-Learning/
│
├── Fashion_MNIST_VGG16.ipynb
├── README.md
└── requirements.txt
```

## 🎯 Key Concepts

This project demonstrates:

* Transfer Learning
* Pretrained Models
* VGG16
* Custom PyTorch Dataset
* DataLoader
* Image Transformations
* Freezing Model Layers
* Custom Classification Head
* Model Training
* Model Evaluation

## 👨‍💻 Author

**Shayan Ahmed**

AI / Machine Learning Engineer

📧 [iamshayanjaved@gmail.com](mailto:iamshayanjaved@gmail.com)
